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

"""Auto-bootstrap IntelliFold-v2 weights for `intellifold predict`.

On first use for a given model directory, download the **already-converted**
AF3-JAX weights (`intellifold_v2.bin.zst`) and Fourier embedding
(`intellifold_v2_fourier.npz`) from Hugging Face into it; later runs find them and
load directly. No PyTorch, no conversion step (the files are pre-converted on the
Hub). Set `HF_ENDPOINT` (e.g. `hf-mirror.com`) to use a mirror.

(To convert your own `.pt` checkpoint instead, see `intellifold.convert` /
`convert_ifv2_to_jax.py`, which needs torch: `pip install '.[convert]'`.)
"""

from __future__ import annotations

import os
import sys
import urllib.request

_HF_REPO = 'intelligenAI/intellifold'
_BIN_NAME = 'intellifold_v2.bin.zst'           # AF3 loads any single *.bin.zst in --model-dir
_FOURIER_NAME = 'intellifold_v2_fourier.npz'


def _hf_url(filename: str) -> str:
  endpoint = os.environ.get('HF_ENDPOINT', 'huggingface.co')
  return f'https://{endpoint}/{_HF_REPO}/resolve/main/{filename}'


def _download(filename: str, dst: str, log) -> None:
  url = _hf_url(filename)
  log(f'[intellifold] downloading {filename}\n    {url}\n    -> {dst}')
  tmp = dst + '.part'

  def _hook(blocks, bsize, total):
    if total and total > 0:
      done = blocks * bsize
      pct = min(100.0, 100.0 * done / total)
      sys.stderr.write(f'\r  {pct:5.1f}%  ({done // (1 << 20)} / {total // (1 << 20)} MB)')
      sys.stderr.flush()

  urllib.request.urlretrieve(url, tmp, reporthook=_hook)
  sys.stderr.write('\n')
  os.replace(tmp, dst)


def ensure_weights(model_dir: str, log=print) -> tuple[str, str]:
  """Make `{model_dir}/intellifold_v2.bin.zst` and `intellifold_v2_fourier.npz`
  exist, downloading the pre-converted files from Hugging Face on first use.
  Returns their absolute paths. No torch / no conversion required.
  """
  model_dir = os.path.abspath(model_dir)
  os.makedirs(model_dir, exist_ok=True)
  bin_path = os.path.join(model_dir, _BIN_NAME)
  fourier_path = os.path.join(model_dir, _FOURIER_NAME)

  if os.path.exists(bin_path) and os.path.exists(fourier_path):
    log(f'[intellifold] using weights in {model_dir}')
    return bin_path, fourier_path

  for filename, dst in ((_BIN_NAME, bin_path), (_FOURIER_NAME, fourier_path)):
    if not os.path.exists(dst):
      _download(filename, dst, log)
  log(f'[intellifold] weights ready in {model_dir} ({_BIN_NAME} + {_FOURIER_NAME})')
  return bin_path, fourier_path
