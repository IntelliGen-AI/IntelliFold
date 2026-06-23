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
