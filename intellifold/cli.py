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

"""`intellifold` command-line entry point.

Usage
-----
  intellifold predict INPUT [--model-dir model_v2] [--gpus all] \\
      [--output-dir out] [--flash cudnn] [-- <extra AlphaFold 3 flags>]

On first use the IntelliFold-v2 weights are auto-downloaded from Hugging Face and
converted into --model-dir (default ./model_v2); later runs just load them.

INPUT is either a single AF3 fold-input JSON file, or a directory of JSON files
(batch -> AF3 --input_dir). Option order is free (the input may come before or
after the options). Anything after a literal `--` is passed verbatim to AF3
(e.g. `--norun_data_pipeline`, `--db_dir=...`, `--num_diffusion_samples=1`).

This is a thin wrapper around the vendored, patched AlphaFold 3 `run_jax_inference.py`:
it sets the two IntelliFold env switches (INTFOLD_FULLFAT, INTFOLD_FOURIER), picks
IntelliFold-friendly defaults (cudnn attention), then hands off to AF3's own main.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

from intellifold import __version__

_EPILOG = textwrap.dedent("""\
    examples:
      # simplest: weights auto-download+convert on first run (into ./model_v2):
      intellifold predict fold_input.json --output-dir out -- --norun_data_pipeline

      # batch a directory across every GPU on the machine:
      intellifold predict ./my_inputs/ --gpus all --output-dir out -- --norun_data_pipeline

      # let AF3 build features first (needs the sequence databases):
      intellifold predict fold_input.json -- --db_dir=/path/to/databases

    notes:
      * weights: on first use they are downloaded from Hugging Face and converted
        into --model-dir (default ./model_v2). The one-time conversion needs torch
        (pip install 'intellifold[convert]'); set HF_ENDPOINT for a mirror.
      * attention kernel: cudnn (default) and triton are both flash kernels (Ampere+).
        xla is the non-flash reference (portable) but materialises the N×N attention
        matrix, so it can OOM on the largest complexes.
      * a persistent XLA compilation cache is ON by default
        (~/.cache/intellifold/jax_compilation): the first run of each input size
        compiles slowly, later runs of the same size skip it. --no-cache to disable.
    """)


def _build_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
      prog='intellifold',
      description='Run IntelliFold-v2 weights on the AlphaFold 3 JAX engine.',
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=_EPILOG,
  )
  p.add_argument('--version', action='version', version=f'intellifold {__version__}')
  sub = p.add_subparsers(dest='command', metavar='COMMAND')

  pred = sub.add_parser(
      'predict',
      help='predict structure(s) for an AF3-format JSON file or a directory of them',
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=_EPILOG,
  )
  pred.add_argument('input', metavar='INPUT',
                    help='an AF3 fold-input JSON file, OR a directory of JSON files (batch)')
  pred.add_argument('--model-dir', default='model_v2',
                    help='directory for the IntelliFold-v2 weights '
                         '(intellifold_v2.bin.zst + intellifold_v2_fourier.npz). On first use it is '
                         'auto-created and the pre-converted weights are downloaded from Hugging '
                         'Face into it; later runs just load them. Default: ./model_v2')
  pred.add_argument('--fourier', default=os.environ.get('INTFOLD_FOURIER'),
                    help="path to the Fourier npz (default: "
                         "<model-dir>/intellifold_v2_fourier.npz, from the auto-conversion)")
  pred.add_argument('--output-dir', default='out', help='where to write predictions (default: ./out)')
  pred.add_argument('--flash', choices=['cudnn', 'xla', 'triton'], default='cudnn',
                    help='flash-attention implementation (default: cudnn)')
  pred.add_argument('--cache-dir',
                    default=os.path.join(
                        os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')),
                        'intellifold', 'jax_compilation'),
                    help='persistent XLA compilation cache directory (on by default). The first '
                         'run of each input size compiles slowly (~minutes); later runs of the '
                         'same size reuse the cache and skip compilation. '
                         'Default: ~/.cache/intellifold/jax_compilation')
  pred.add_argument('--no-cache', dest='cache_dir', action='store_const', const=None,
                    help='disable the persistent compilation cache (always recompile)')
  pred.add_argument('--gpus', default=None,
                    help="use multiple GPUs on this machine: 'all' or a comma list like '0,2,3'. "
                         "INPUT must be a directory; one worker process per GPU pulls targets from "
                         "a local work queue under <output-dir>/.queue, self-balancing and "
                         "resumable (rerun to continue after a crash). Omit for a single-GPU run.")
  pred.add_argument('--retry-failed', action='store_true',
                    help='with --gpus, also re-attempt targets previously marked .failed')
  pred.add_argument('--reset-stale', action='store_true',
                    help='with --gpus, clear orphaned .claim locks from crashed workers before '
                         'starting (only safe when no other workers are running this queue)')
  pred.add_argument('--full-fat', dest='full_fat', action='store_true', default=True,
                    help='use IntelliFold-v2 (full_fat) config dims [default]')
  pred.add_argument('--no-full-fat', dest='full_fat', action='store_false',
                    help='run the stock AF3 config instead (for stock af3.bin)')
  return p


def _require_engine() -> None:
  """Fail fast with actionable guidance if the AlphaFold 3 JAX engine is absent.

  IntelliFold runs on AF3's JAX engine, which is NOT on PyPI -- it must be built
  from the Apache-2.0 source vendored in third_party/alphafold3/. `pip install
  intellifold` only installs this wrapper, so check the engine is importable
  before doing any real work (e.g. downloading the multi-GB weights).
  """
  import importlib.util
  if importlib.util.find_spec('alphafold3') is None:
    sys.exit(
        "error: the AlphaFold 3 JAX engine (module 'alphafold3') is not installed.\n"
        '  IntelliFold runs on AF3\'s JAX engine, which is not on PyPI and must be\n'
        '  built from the vendored Apache-2.0 source:\n'
        '      pip install ./third_party/alphafold3 && build_data\n'
        '  See the Setup section of the README.')


def _run_predict(args, af3_extra) -> int:
  _require_engine()  # fail fast (before any download) if alphafold3 is missing

  # Auto-bootstrap the IntelliFold-v2 weights: on first use, download the
  # pre-converted intellifold_v2.bin.zst + intellifold_v2_fourier.npz from Hugging
  # Face into --model-dir (no torch / no conversion); later runs just load them.
  # (full_fat only; --no-full-fat expects a user-supplied stock af3.bin.zst.)
  if args.full_fat:
    from intellifold.weights import ensure_weights
    _bin, _fourier = ensure_weights(args.model_dir)
    if not args.fourier:
      args.fourier = _fourier
  elif not args.fourier:
    sys.exit('error: --fourier is required with --no-full-fat (no auto-conversion).')

  # Multi-GPU / multi-node batch: become an orchestrator that spawns one
  # single-GPU worker per GPU, all sharing a work queue under <output-dir>/.queue.
  if args.gpus is not None:
    return _run_orchestrator(args, af3_extra)

  # IntelliFold switches consumed by intellifold.patches inside run_jax_inference.
  if args.full_fat:
    os.environ['INTFOLD_FULLFAT'] = '1'
    os.environ['INTFOLD_FOURIER'] = os.path.abspath(args.fourier)

  # Persistent XLA compilation cache (on by default): the first run of each input
  # size pays the slow JIT compile, later runs of the same size reuse it. Skip if
  # the user disabled it (--no-cache) or passed their own --jax_compilation_cache_dir.
  if args.cache_dir and not any(a.startswith('--jax_compilation_cache_dir') for a in af3_extra):
    os.makedirs(args.cache_dir, exist_ok=True)
    af3_extra = [*af3_extra, f'--jax_compilation_cache_dir={args.cache_dir}']
    print(f'[intellifold] JAX compilation cache: {args.cache_dir}')

  # A directory -> AF3 batch mode (--input_dir); a file -> --json_path.
  if os.path.isdir(args.input):
    input_flag = f'--input_dir={args.input}'
  else:
    input_flag = f'--json_path={args.input}'

  # Assemble AlphaFold 3's own (absl) argv and hand off. af3_extra is whatever the
  # user put after `--`, passed through verbatim.
  af3_argv = [
      'intellifold-predict',
      input_flag,
      f'--model_dir={args.model_dir}',
      f'--output_dir={args.output_dir}',
      f'--flash_attention_implementation={args.flash}',
      *af3_extra,
  ]

  # Deferred import: keeps `intellifold --help` light (no jax) and only loads the
  # heavy AF3 stack when actually predicting.
  from absl import app
  from intellifold import run_jax_inference

  app.run(run_jax_inference.main, argv=af3_argv)
  return 0


def _run_orchestrator(args, af3_extra) -> int:
  """Single-machine multi-GPU batch driver.

  Spawns one single-GPU worker per GPU; every worker claims targets from
  <output-dir>/.queue, so the directory is divided across this machine's GPUs
  with no overlap, self-balancing, and the run resumes if restarted.
  """
  from intellifold import parallel

  if not os.path.isdir(args.input):
    sys.exit('error: --gpus requires INPUT to be a directory of JSON files.')
  gpus = parallel.resolve_gpus(args.gpus)
  if not gpus:
    sys.exit(f'error: no GPUs resolved from --gpus {args.gpus!r}.')

  queue_dir = os.path.abspath(os.path.join(args.output_dir, '.queue'))
  os.makedirs(queue_dir, exist_ok=True)
  q = parallel.WorkQueue(args.input, queue_dir, retry_failed=args.retry_failed)
  if args.reset_stale:
    n = q.reset_stale()
    print(f'[orchestrator] cleared {n} stale .claim lock(s)')
  s = q.stats()
  print(f'[orchestrator] {s["total"]} targets | {s["done"]} done, {s["failed"]} failed, '
        f'{s["remaining"]} to go | {len(gpus)} local GPU(s): {gpus}')
  if s['remaining'] == 0:
    print('[orchestrator] nothing to do (queue already drained).')
    return 0

  # Each worker is this same CLI in classic single-GPU mode, with the work-queue
  # flag appended so run_jax_inference pulls from the shared queue instead of looping
  # the whole directory. CUDA_VISIBLE_DEVICES pins it to one GPU.
  worker_extra = [*af3_extra, f'--work_queue_dir={queue_dir}']
  if args.retry_failed:
    worker_extra.append('--work_queue_retry_failed')
  worker_argv = [
      sys.executable, '-m', 'intellifold.cli', 'predict', args.input,
      '--model-dir', args.model_dir,
      '--output-dir', args.output_dir,
      '--flash', args.flash,
  ]
  if args.fourier:
    worker_argv += ['--fourier', args.fourier]
  if not args.full_fat:
    worker_argv += ['--no-full-fat']
  worker_argv += ['--cache-dir', args.cache_dir] if args.cache_dir else ['--no-cache']
  worker_argv += ['--', *worker_extra]

  rc = parallel.spawn_workers(
      worker_argv, gpus, queue_dir,
      log_dir=os.path.join(args.output_dir, 'worker_logs'))
  s = q.stats()
  print(f'[orchestrator] finished | {s["done"]} done, {s["failed"]} failed, '
        f'{s["remaining"]} remaining (rerun with the same --output-dir to resume)')
  return rc


def main(argv=None) -> int:
  if argv is None:
    argv = sys.argv[1:]
  argv = list(argv)
  # Split off AF3 pass-through flags at the first standalone `--`. This avoids
  # argparse.REMAINDER, which would otherwise swallow our own options whenever the
  # INPUT comes before them. With this, option/input order is free.
  if '--' in argv:
    i = argv.index('--')
    own, af3_extra = argv[:i], argv[i + 1:]
  else:
    own, af3_extra = argv, []

  parser = _build_parser()
  args = parser.parse_args(own)
  if args.command == 'predict':
    return _run_predict(args, af3_extra)
  parser.print_help()
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
