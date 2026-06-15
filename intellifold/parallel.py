# Copyright 2026 IntelliGen-AI and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-machine multi-GPU batch execution for `intellifold predict`.

AlphaFold 3's inference script processes a directory of inputs *sequentially on a
single GPU*. To use every GPU on the machine we put a small **work queue** in
front of it: one worker process per GPU, each claiming targets from a queue
directory (under the output dir) by an atomic `mkdir`, so

  * no two GPU workers ever take the same target (the mkdir is the lock),
  * load balances itself — a worker that finishes a target grabs the next one,
    so big targets don't stall the others, and
  * a re-run skips finished targets (`.done`) and already-tried ones (`.failed`),
    i.e. it resumes after a crash.

Each worker is a separate process (not a thread), so one target running out of
memory cannot take down the workers on the other GPUs.

This module is deliberately dependency-free (no jax / alphafold3 import) so the
CLI can import it cheaply; the heavy `load_fn` is injected by the worker.
"""

from __future__ import annotations

import glob
import os
import pathlib
import signal
import socket
import subprocess
import sys
from typing import Callable, Iterator, Sequence


# --------------------------------------------------------------------------- #
#  Target discovery
# --------------------------------------------------------------------------- #
def list_targets(input_dir: str) -> list[str]:
  """Every AF3 fold-input JSON in `input_dir` (sorted, sidecars excluded)."""
  files = sorted(glob.glob(os.path.join(input_dir, '*.json')))
  # Skip editor/backup sidecars that happen to live alongside the inputs.
  return [f for f in files if not f.endswith(('.backup', '.bak', '.tmp'))]


def _stem(json_path: str) -> str:
  return pathlib.Path(json_path).stem


# --------------------------------------------------------------------------- #
#  The shared work queue
# --------------------------------------------------------------------------- #
class WorkQueue:
  """A claim/done/failed work queue keyed by input-file stem.

  Uses atomic `mkdir` as the lock, so the per-GPU worker processes never collide.

  Markers in `queue_dir`:
    <stem>.claim  -- a directory (atomic mkdir) held while a worker runs it
    <stem>.done   -- written after the target finished successfully
    <stem>.failed -- written if the target raised (so we don't retry forever)
  """

  def __init__(self, input_dir: str, queue_dir: str, retry_failed: bool = False):
    self.input_dir = input_dir
    self.queue_dir = queue_dir
    self.retry_failed = retry_failed
    os.makedirs(queue_dir, exist_ok=True)

  def _p(self, stem: str, ext: str) -> str:
    return os.path.join(self.queue_dir, stem + ext)

  def _try_claim(self, stem: str) -> bool:
    if os.path.exists(self._p(stem, '.done')):
      return False
    if os.path.exists(self._p(stem, '.failed')) and not self.retry_failed:
      return False
    try:
      os.mkdir(self._p(stem, '.claim'))  # atomic on POSIX & NFS
      return True
    except FileExistsError:
      return False                        # held by another worker (or stale)

  def _release(self, stem: str) -> None:
    try:
      os.rmdir(self._p(stem, '.claim'))
    except OSError:
      pass

  def mark_done(self, stem: str) -> None:
    with open(self._p(stem, '.done'), 'w') as fh:
      fh.write('ok\n')
    self._release(stem)

  def mark_failed(self, stem: str, err: object) -> None:
    with open(self._p(stem, '.failed'), 'w') as fh:
      fh.write(f'{type(err).__name__}: {err}\n')
    self._release(stem)

  def claim_iter(
      self, load_fn: Callable[[pathlib.Path], Iterator]
  ) -> Iterator[tuple[str, object]]:
    """Yield `(stem, fold_input)` for every target this worker claims.

    `load_fn` is `folding_input.load_fold_inputs_from_path`. The consumer must
    call `mark_done(stem)` on success or `mark_failed(stem, err)` on error before
    requesting the next item, so the claim is released exactly once.
    """
    for jp in list_targets(self.input_dir):
      stem = _stem(jp)
      if not self._try_claim(stem):
        continue
      print(f'[queue] claimed {stem}', flush=True)
      try:
        loaded = list(load_fn(pathlib.Path(jp)))
      except Exception as e:                                   # noqa: BLE001
        print(f'[queue] FAILED to load {stem}: {e}', flush=True)
        self.mark_failed(stem, e)
        continue
      for fold_input in loaded:
        yield stem, fold_input

  def stats(self) -> dict[str, int]:
    total = len(list_targets(self.input_dir))
    done = len(glob.glob(os.path.join(self.queue_dir, '*.done')))
    failed = len(glob.glob(os.path.join(self.queue_dir, '*.failed')))
    claimed = len(glob.glob(os.path.join(self.queue_dir, '*.claim')))
    return {
        'total': total, 'done': done, 'failed': failed,
        'in_progress': claimed, 'remaining': total - done - failed,
    }

  def reset_stale(self) -> int:
    """Remove `.claim` markers with no `.done`/`.failed` (crash recovery).

    Only safe to call when no workers are running against this queue.
    """
    n = 0
    for claim in glob.glob(os.path.join(self.queue_dir, '*.claim')):
      stem = pathlib.Path(claim).stem
      if not (os.path.exists(self._p(stem, '.done'))
              or os.path.exists(self._p(stem, '.failed'))):
        try:
          os.rmdir(claim); n += 1
        except OSError:
          pass
    return n


# --------------------------------------------------------------------------- #
#  Local (single-node) multi-GPU orchestration
# --------------------------------------------------------------------------- #
def resolve_gpus(spec: str | None) -> list[int]:
  """'all' / '0,1,2' / None -> a list of GPU indices on this node."""
  if spec is None:
    return []
  if spec.strip().lower() == 'all':
    try:
      out = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True,
                           check=True).stdout
      n = sum(1 for line in out.splitlines() if line.startswith('GPU '))
      return list(range(n))
    except Exception:                                          # noqa: BLE001
      vis = os.environ.get('CUDA_VISIBLE_DEVICES')
      if vis:
        return list(range(len([x for x in vis.split(',') if x != ''])))
      raise SystemExit('error: could not enumerate GPUs for --gpus all')
  return [int(x) for x in spec.replace(' ', '').split(',') if x != '']


def spawn_workers(
    worker_argv: Sequence[str],
    gpus: Sequence[int],
    queue_dir: str,
    log: Callable[[str], None] = print,
    log_dir: str | None = None,
) -> int:
  """Launch one worker process per GPU; each pins to its GPU via
  CUDA_VISIBLE_DEVICES and pulls from the shared queue. Returns the worst
  child return code. Ctrl-C terminates the whole pool.

  If `log_dir` is set, each worker's stdout/stderr is redirected to its own file
  `<log_dir>/<hostname>_gpu<g>.log` (one sequential log per GPU — keeps each
  target's AF3 output, incl. token count and per-seed timing, un-interleaved).
  """
  host = socket.gethostname()
  logfiles = []
  if log_dir:
    os.makedirs(log_dir, exist_ok=True)
  procs: list[tuple[int, subprocess.Popen]] = []
  for g in gpus:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
    out = None
    if log_dir:
      out = open(os.path.join(log_dir, f'{host}_gpu{g}.log'), 'a')
      logfiles.append(out)
    p = subprocess.Popen(list(worker_argv), env=env,
                         stdout=out, stderr=subprocess.STDOUT if out else None)
    procs.append((g, p))
    log(f'[orchestrator] GPU {g} -> worker pid {p.pid}'
        + (f' (log: {host}_gpu{g}.log)' if log_dir else ''))
  log(f'[orchestrator] {len(procs)} workers up; queue: {queue_dir}')

  rc = 0
  try:
    for g, p in procs:
      r = p.wait()
      log(f'[orchestrator] GPU {g} worker exited rc={r}')
      rc = rc or r
  except KeyboardInterrupt:
    log('[orchestrator] interrupted -- terminating workers...')
    for _, p in procs:
      p.send_signal(signal.SIGINT)
    for _, p in procs:
      try:
        p.wait(timeout=30)
      except subprocess.TimeoutExpired:
        p.kill()
    rc = 130
  finally:
    for fh in logfiles:
      try:
        fh.close()
      except OSError:
        pass
  return rc
