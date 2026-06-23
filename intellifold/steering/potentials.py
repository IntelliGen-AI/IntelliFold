"""Differentiable steering potentials in JAX.

We define only the scalar potential *energies*; the guidance loop differentiates
them with ``jax.grad`` (no hand-derived analytic gradients needed). Every
constant buffer (bond / angle / clash margins, chirality / stereo / planar angle
thresholds) is baked into the per-constraint ``lower`` / ``upper`` bounds on the
host, so the JAX side reduces to two flat-bottom kernels: one on inter-atom
distances, one on dihedral angles.

All potentials operate on a flat atom tensor ``coords`` of shape
``[N_atoms, 3]`` where ``N_atoms = num_tokens * atoms_per_token`` and the atom
at (token ``t``, slot ``a``) lives at flat index ``t * atoms_per_token + a``.
Index arrays select into that flat axis.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


_EPS = 1e-8


def flat_bottom_energy(value, lower, upper, k=1.0):
    """Hooke energy with a flat bottom between ``lower`` and ``upper``.

    Zero inside ``[lower, upper]``; grows linearly with slope ``k`` outside.
    ``lower`` / ``upper`` may contain +/- inf to leave a side unbounded.
    Returns the summed scalar energy.
    """
    below = jnp.where(jnp.isneginf(lower), 0.0, jnp.maximum(lower - value, 0.0))
    above = jnp.where(jnp.isposinf(upper), 0.0, jnp.maximum(value - upper, 0.0))
    return k * jnp.sum(below + above)


def pair_distance(coords, index):
    """Distances between atom ``index[0]`` and atom ``index[1]``. -> [P]."""
    r = coords[index[0]] - coords[index[1]]
    return jnp.sqrt(jnp.sum(r * r, axis=-1) + _EPS)


def dihedral_angle(coords, index):
    """Signed dihedral phi defined by four indexed atoms. -> [M] in [-pi, pi].

    Uses the ``atan2`` formulation rather than ``arccos``. ``arccos`` has a
    derivative ``-1/sqrt(1 - u^2)`` that blows up to +/-inf exactly when the
    dihedral approaches 0 or pi -- which is precisely where planar / stereo
    constraints push -- so differentiating it through ``jax.grad`` yields NaNs.
    ``atan2`` is smooth across the whole circle and gives well-behaved grads.
    """
    p0 = coords[index[0]]
    p1 = coords[index[1]]
    p2 = coords[index[2]]
    p3 = coords[index[3]]

    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1n = b1 / (jnp.linalg.norm(b1, axis=-1, keepdims=True) + _EPS)

    # Components of b0 and b2 perpendicular to the central bond b1.
    v = b0 - jnp.sum(b0 * b1n, axis=-1, keepdims=True) * b1n
    w = b2 - jnp.sum(b2 * b1n, axis=-1, keepdims=True) * b1n

    x = jnp.sum(v * w, axis=-1)
    y = jnp.sum(jnp.cross(b1n, v) * w, axis=-1)
    return jnp.arctan2(y, x)


# ----------------------------------------------------------------------------
# Per-group energies. Each takes the flat ``coords`` and the group's host-built
# index / bound arrays, returning a scalar energy. Empty groups (M == 0) give 0.
# ----------------------------------------------------------------------------


def distance_energy(coords, index, lower, upper):
    """Flat-bottom on pairwise distances (PoseBusters bounds, connections, VDW)."""
    if index.shape[1] == 0:
        return jnp.zeros((), dtype=coords.dtype)
    return flat_bottom_energy(pair_distance(coords, index), lower, upper)


def signed_dihedral_energy(coords, index, lower, upper):
    """Flat-bottom on signed dihedral angles (chirality)."""
    if index.shape[1] == 0:
        return jnp.zeros((), dtype=coords.dtype)
    return flat_bottom_energy(dihedral_angle(coords, index), lower, upper)


def abs_dihedral_energy(coords, index, lower, upper):
    """Flat-bottom on |dihedral| (stereo bonds, planar improper)."""
    if index.shape[1] == 0:
        return jnp.zeros((), dtype=coords.dtype)
    return flat_bottom_energy(jnp.abs(dihedral_angle(coords, index)), lower, upper)


def com_distance_energy(coords, weight, pairs, lower):
    """Flat-bottom repulsion between symmetric chains' centres of mass.

    ``weight`` is a ``[C, N]`` per-chain averaging matrix (rows sum to 1 over a
    chain's real atoms), so ``weight @ coords`` gives each chain's COM. ``pairs``
    selects the symmetric chain pairs; ``lower`` is the (time-scheduled) minimum
    COM separation. Mirrors SymmetricChainCOMPotential.
    """
    if pairs.shape[1] == 0:
        return jnp.zeros((), dtype=coords.dtype)
    com = weight @ coords  # [C, 3]
    d = jnp.linalg.norm(com[pairs[0]] - com[pairs[1]], axis=-1)  # [M]
    lower = jnp.broadcast_to(jnp.asarray(lower, coords.dtype), d.shape)
    upper = jnp.full_like(d, jnp.inf)
    return flat_bottom_energy(d, lower, upper)


# Maps a group's variable kind to its energy function. The host tags each group
# with one of these names.
ENERGY_FNS = {
    "distance": distance_energy,
    "dihedral": signed_dihedral_energy,
    "abs_dihedral": abs_dihedral_energy,
}
