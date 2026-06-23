"""Diffusion steering / guidance for the IntelliFold JAX engine.

Boltz-style inference-time physical-guidance steering for the AlphaFold 3 JAX
sampler. Default OFF; enabled per-run via the ``--steering`` flag (see
``intellifold/run_jax_inference.py``).

Two pieces:
* ``rdkit_features.build_steering_features`` - host-side (CPU) RDKit constraint
  extraction from a featurised example, returning grouped numpy arrays.
* ``guidance.apply_guidance`` - JAX gradient guidance applied inside the
  diffusion sampler's denoising loop.
"""

from intellifold.steering.guidance import apply_guidance, default_groups
from intellifold.steering.rdkit_features import build_steering_features

# Keys this package attaches to the featurised example dict; the model picks
# them up before converting the dict to a typed Batch.
STEERING_KEY_PREFIX = "steering__"

__all__ = [
    "apply_guidance",
    "default_groups",
    "build_steering_features",
    "STEERING_KEY_PREFIX",
]
