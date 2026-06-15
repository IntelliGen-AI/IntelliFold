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

"""Runtime patches that let the stock AlphaFold 3 JAX network load IntelliFold-v2.

IntelliFold-v2 is architecturally identical to AlphaFold 3 except for (a) a few
hidden sizes (the "full_fat" preset) and (b) its own trained Fourier
noise-embedding. The official AF3 code can therefore run v2 with two small
changes, applied here at runtime instead of by hand-editing installed files:

  * `apply_fourier_patch()` rebinds
    `alphafold3.model.network.noise_level_embeddings.noise_embeddings` so it reads
    v2's Fourier weight/bias from the `.npz` produced by
    `convert_ifv2_to_jax.py` (path in $INTFOLD_FOURIER). This avoids the
    "pip-install gotcha" of editing the *installed* site-packages module;
    monkey-patching the module attribute does the same thing without touching
    site-packages.

  * `widen_config_full_fat(config)` widens the model config (c_z/c_m/c_t and a
    few head counts) to v2's dimensions.

Both functions are **env-gated and idempotent**: with neither $INTFOLD_FOURIER
nor $INTFOLD_FULLFAT set they are no-ops, so importing this module never changes
stock AlphaFold 3 behaviour.
"""

from __future__ import annotations

import os

_FOURIER_PATCH_APPLIED = False


def apply_fourier_patch(npz_path: str | None = None) -> bool:
  """Inject IntelliFold-v2's trained Fourier noise-embedding into AF3.

  AF3 hardcodes `_WEIGHT`/`_BIAS` in `noise_embeddings()` (derived from
  `PRNGKey(42)`); IntelliFold-v2 trained its own. The converter drops them from
  the weight file, so they must be fed back in at runtime or the diffusion head
  sees the wrong embedding and the structures explode (CA-CA ~9.8 A) despite a
  high ipTM.

  Args:
    npz_path: path to `intfold_fourier.npz` (keys `weight`, `bias`). Defaults to
      the $INTFOLD_FOURIER environment variable.

  Returns:
    True if the patch was applied, False if it was a no-op (no path given).
  """
  global _FOURIER_PATCH_APPLIED
  npz_path = npz_path or os.environ.get('INTFOLD_FOURIER')
  if not npz_path:
    return False  # unpatched AF3 path -- keep the official _WEIGHT/_BIAS
  if _FOURIER_PATCH_APPLIED:
    return True

  import numpy as np
  from alphafold3.model.network import noise_level_embeddings as nle

  if not os.path.exists(npz_path):
    raise FileNotFoundError(
        f'INTFOLD_FOURIER points at a missing file: {npz_path!r}. '
        'Produce it with convert_ifv2_to_jax.py.'
    )
  data = np.load(npz_path)
  weight_np = np.asarray(data['weight'], dtype=np.float32)
  bias_np = np.asarray(data['bias'], dtype=np.float32)

  def noise_embeddings(sigma_scaled_noise_level):
    """IntelliFold-v2 Fourier embedding (same math as AF3, v2's weight/bias)."""
    import jax.numpy as jnp

    transformed_noise_level = (1 / 4) * jnp.log(sigma_scaled_noise_level)
    weight = jnp.asarray(weight_np, dtype=jnp.float32)
    bias = jnp.asarray(bias_np, dtype=jnp.float32)
    embeddings = transformed_noise_level[..., None] * weight + bias
    return jnp.cos(2 * jnp.pi * embeddings)

  # diffusion_head.py calls `noise_level_embeddings.noise_embeddings(...)` as a
  # module attribute, so rebinding the attribute is sufficient.
  nle.noise_embeddings = noise_embeddings
  _FOURIER_PATCH_APPLIED = True
  print(
      f'[intellifold] Fourier embedding patched from {npz_path} '
      f'(weight{weight_np.shape}, bias{bias_np.shape})',
      flush=True,
  )
  return True


# IntelliFold-v2 "full_fat" preset. Everything not listed is identical to the
# public AF3 config (c_s=384/16 heads, c_token=768/16, c_atom=128/4; block counts
# 48 pairformer / 24 diffusion-transformer / 4 msa / 2 template / 4 confidence).
_FULLFAT_OVERRIDES = {
    'evoformer.pair_channel': 512,                                   # c_z
    'evoformer.msa_channel': 256,                                    # c_m
    'evoformer.template.num_channels': 256,                          # c_t
    'heads.diffusion.conditioning.pair_channel': 512,
    'evoformer.msa_stack.pair_attention.num_head': 8,               # no_heads_pair
    'evoformer.pairformer.pair_attention.num_head': 8,
    'evoformer.template.template_stack.pair_attention.num_head': 8,
    'heads.confidence.pairformer.pair_attention.num_head': 8,
}


def _set_dotted(obj, dotted: str, value) -> None:
  """Set obj.a.b.c = value, raising a clear error if the path drifted."""
  *parents, leaf = dotted.split('.')
  node = obj
  for p in parents:
    if not hasattr(node, p):
      raise AttributeError(
          f'full_fat config path {dotted!r} not found at {p!r}. The AlphaFold 3 '
          'config layout may have changed in this version -- re-check the '
          'overrides against this AF3 release before trusting the output.'
      )
    node = getattr(node, p)
  if not hasattr(node, leaf):
    raise AttributeError(
        f'full_fat config path {dotted!r} not found (missing {leaf!r}); '
        'AF3 config layout drift -- re-validate the overrides.'
    )
  setattr(node, leaf, value)


def widen_config_full_fat(config) -> bool:
  """Widen `config` in place to IntelliFold-v2 (full_fat) dims if $INTFOLD_FULLFAT=1.

  Returns True if applied, False if it was a no-op.
  """
  if os.environ.get('INTFOLD_FULLFAT') != '1':
    return False
  for dotted, value in _FULLFAT_OVERRIDES.items():
    _set_dotted(config, dotted, value)
  print('[intellifold] model config widened to full_fat (IntelliFold-v2)', flush=True)
  return True
