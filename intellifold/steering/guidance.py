"""Gradient-based diffusion guidance (physical-guidance steering).

At each denoising step we run ``num_gd_steps`` of gradient descent on the sum of
the active potentials and add the accumulated displacement to the model's x0
prediction. Potential gradients come from ``jax.grad``; the dihedral uses the
``atan2`` form so its gradient is finite everywhere, so no displacement clipping
is needed.

The per-group guidance weights, intervals and schedules cover the Boltz-style
physical-guidance set: PoseBusters bounds, Connections, VDW overlap,
symmetric-chain COM, chirality, stereo, and planar bonds.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, NamedTuple

import jax
import jax.numpy as jnp

from intellifold.steering import potentials as P


# ----------------------------------------------------------------------------
# Schedules (functions of the normalised diffusion time t in [0, 1], where
# t = 1 - step / num_steps).
# ----------------------------------------------------------------------------


def constant(value: float) -> Callable[[jnp.ndarray], jnp.ndarray]:
    return lambda t: jnp.asarray(value, jnp.float32)


def piecewise_step(thresholds: List[float], values: List[float]):
    """values[i] for the smallest i with t <= thresholds[i], else values[-1]."""

    def fn(t):
        out = jnp.asarray(values[-1], jnp.float32)
        for thr, val in zip(reversed(thresholds), reversed(values[:-1])):
            out = jnp.where(t <= thr, jnp.asarray(val, jnp.float32), out)
        return out

    return fn


def exponential_interp(start: float, end: float, alpha: float):
    """Boltz ExponentialInterpolation: start..end as t goes 0->1."""
    denom = math.exp(alpha) - 1.0

    def fn(t):
        if alpha == 0:
            return jnp.asarray(start + (end - start) * t, jnp.float32)
        return start + (end - start) * (jnp.exp(alpha * t) - 1.0) / denom

    return fn


class Group(NamedTuple):
    """One flat-bottom potential family: variable kind, weight schedule, GD interval."""

    name: str
    kind: str  # "distance" | "dihedral" | "abs_dihedral"
    weight_fn: Callable[[jnp.ndarray], jnp.ndarray]
    interval: int


# Default physical-guidance groups (Boltz-style weights / intervals).
def default_groups() -> List[Group]:
    return [
        Group("posebusters", "distance", constant(0.01), 1),
        Group("connections", "distance", constant(0.15), 1),
        Group("chiral", "dihedral", constant(0.1), 1),
        Group("stereo", "abs_dihedral", constant(0.05), 1),
        Group("planar", "abs_dihedral", constant(0.05), 1),
        Group("vdw", "distance", piecewise_step([0.4], [0.125, 0.0]), 5),
    ]


# Symmetric-chain COM potential is handled separately (it needs per-chain COMs
# and a time-scheduled buffer rather than host-baked bounds).
_SYMCOM_WEIGHT = constant(0.5)
_SYMCOM_INTERVAL = 4
_SYMCOM_BUFFER = exponential_interp(start=1.0, end=5.0, alpha=-2.0)


def _group_arrays(steering: Dict[str, jnp.ndarray], name: str):
    """Return (index, lower, upper) for a group, or None if absent/empty."""
    idx = steering.get(f"{name}_index")
    if idx is None or idx.shape[1] == 0:
        return None
    return idx, steering[f"{name}_lower"], steering[f"{name}_upper"]


def apply_guidance(
    positions_denoised: jnp.ndarray,
    atom_mask: jnp.ndarray,
    steering: Dict[str, jnp.ndarray],
    t: jnp.ndarray,
    is_last_step: jnp.ndarray,
    num_gd_steps: int,
    weight_scale: float,
    groups: List[Group],
) -> jnp.ndarray:
    """Gradient-guide a single sample's x0 prediction.

    Args:
      positions_denoised: ``[num_tokens, atoms_per_token, 3]`` x0 prediction.
      atom_mask: ``[num_tokens, atoms_per_token]`` bool; padding atoms stay put.
      steering: dict of host-built jnp index / bound arrays per group.
      t: scalar normalised diffusion time for schedule evaluation.
      is_last_step: scalar bool; guidance is skipped on the final step.
      num_gd_steps: gradient-descent iterations per denoising step.
      weight_scale: global multiplier on every group weight (1.0 = faithful).
      groups: active flat-bottom potential groups.

    Returns:
      The guided ``[num_tokens, atoms_per_token, 3]`` positions.
    """
    shape = positions_denoised.shape
    coords = positions_denoised.reshape(-1, 3)
    mask = atom_mask.reshape(-1).astype(coords.dtype)[:, None]

    # Flat-bottom groups that have constraints for this target.
    active = []
    for g in groups:
        arrs = _group_arrays(steering, g.name)
        if arrs is not None:
            active.append((g, arrs))

    sym = None
    if "symcom_pairs" in steering and steering["symcom_pairs"].shape[1] > 0:
        sym = (steering["symcom_weight"], steering["symcom_pairs"])

    if not active and sym is None:
        return positions_denoised

    def energy_at(x, gd_step):
        e = jnp.zeros((), dtype=x.dtype)
        for g, (idx, lo, hi) in active:
            if gd_step % g.interval != 0:
                continue
            w = weight_scale * g.weight_fn(t)
            e = e + w * P.ENERGY_FNS[g.kind](x, idx, lo, hi)
        if sym is not None and gd_step % _SYMCOM_INTERVAL == 0:
            weight, pairs = sym
            w = weight_scale * _SYMCOM_WEIGHT(t)
            e = e + w * P.com_distance_energy(x, weight, pairs, _SYMCOM_BUFFER(t))
        return e

    grad_energy = jax.grad(energy_at, argnums=0)

    # Gradient descent on the accumulated displacement:
    # ``guidance_update -= sum_p gw_p * grad E_p`` over num_gd_steps iterations.
    # A non-finite gradient component is dropped to zero: ``jax.grad`` of
    # ``atan2`` is NaN only at an exactly-collinear (measure-zero) config, so
    # zeroing those keeps the sampler from propagating NaNs. No magnitude clipping.
    guidance_update = jnp.zeros_like(coords)
    for gd_step in range(num_gd_steps):
        g = grad_energy(coords + guidance_update, gd_step)
        g = jnp.where(jnp.isfinite(g), g, 0.0)
        guidance_update = guidance_update - g

    guidance_update = guidance_update * mask
    guidance_update = jnp.where(is_last_step, 0.0, guidance_update)
    return (coords + guidance_update).reshape(shape)
