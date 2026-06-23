"""Host-side steering feature builder (runs on CPU, before the jitted model).

Rebuilds an RDKit ``Mol`` for every ligand residue from the featurised
example's CCD layout, reads off the Boltz/PoseBusters geometry constraints
(distance bounds, chiral centres, E/Z stereo bonds, planar double bonds), and
emits them as plain numpy arrays keyed by potential group.

All constant buffers (bond / angle / clash margins, chirality / stereo / planar
angle thresholds) are baked into per-constraint ``lower`` / ``upper`` bounds
here, so the JAX side only has to evaluate flat-bottom energies. Atom indices
use the dense flat convention ``token_idx * atoms_per_token + slot`` that the
diffusion sampler's ``[num_tokens, atoms_per_token, 3]`` layout flattens to.

Returns ``None`` when nothing constrainable is found (e.g. a protein-only
target), in which case the caller attaches no steering keys and the run is
identical to a vanilla prediction.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional

import numpy as np

from intellifold.steering.constants import ATOMS_PER_TOKEN, NUM_ELEMENTS, VDW_RADII_BY_Z

_LOG = logging.getLogger(__name__)

# Boltz-style physical-guidance buffer defaults.
_BOND_BUFFER = 0.125
_ANGLE_BUFFER = 0.125
_CLASH_BUFFER = 0.10
_CHIRAL_BUFFER = 0.52360
_STEREO_BUFFER = 0.52360
_PLANAR_BUFFER = 0.26180
# VDWOverlapPotential / ConnectionsPotential buffers (constant -> baked here).
# SymmetricChainCOMPotential's buffer is time-scheduled (1->5 A), so it is
# evaluated in JAX rather than baked; the host only emits chain membership.
_VDW_BUFFER = 0.225
_CONNECTIONS_BUFFER = 2.0

_PLANAR_SMARTS = "[C;X3;^2](*)(*)=[C;X3;^2](*)(*)"


def _vdw_radii_table() -> np.ndarray:
    radii = np.full(NUM_ELEMENTS, 2.0, dtype=np.float32)
    radii[1 : 1 + len(VDW_RADII_BY_Z)] = np.asarray(VDW_RADII_BY_Z, dtype=np.float32)
    radii[0] = 0.0
    return radii


# ----------------------------------------------------------------------------
# RDKit mol construction + constraint extraction (one residue at a time)
# ----------------------------------------------------------------------------


def _ccd_mol(ccd, res_name: str):
    """Build an RDKit Mol from the CCD entry, with stereo perceived."""
    from alphafold3.data.tools import rdkit_utils
    import rdkit.Chem as Chem

    entry = ccd.get(res_name)
    if entry is None:
        return None
    try:
        mol = rdkit_utils.mol_from_ccd_cif(
            entry, force_parse=True, sort_alphabetically=False, remove_hydrogens=True
        )
    except Exception as e:  # pragma: no cover - depends on CCD entry
        _LOG.debug("mol_from_ccd_cif failed for %s: %s", res_name, e)
        return None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    try:
        Chem.AssignStereochemistry(
            mol, cleanIt=True, force=True, flagPossibleStereoCenters=True
        )
    except Exception:
        pass
    return mol


def _idx_map(mol, name_to_flat: Dict[str, int]) -> Dict[int, int]:
    """Map mol atom index -> flat IntelliFold atom index by atom name."""
    out = {}
    for ai, atom in enumerate(mol.GetAtoms()):
        if not atom.HasProp("atom_name"):
            continue
        nm = atom.GetProp("atom_name").strip()
        if nm in name_to_flat:
            out[ai] = name_to_flat[nm]
    return out


def _geometry_constraints(mol, idx_map):
    """RDKit bounds matrix -> (pairs[2,P], lo[P], hi[P], is_bond[P], is_angle[P])."""
    import rdkit.Chem as Chem
    from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix

    n = mol.GetNumAtoms()
    if n <= 1:
        return None
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)
    bounds = GetMoleculeBoundsMatrix(
        mol, set15bounds=True, scaleVDW=True, doTriangleSmoothing=True,
        useMacrocycle14config=False,
    )
    bond_set = {tuple(sorted(b)) for b in mol.GetSubstructMatches(Chem.MolFromSmarts("*~*"))}
    angle_set = {
        tuple(sorted([a[0], a[2]]))
        for a in mol.GetSubstructMatches(Chem.MolFromSmarts("*~*~*"))
    }
    pairs, lo, hi, isb, isa = [], [], [], [], []
    iu, ju = np.triu_indices(n, k=1)
    for i, j in zip(iu.tolist(), ju.tolist()):
        if i not in idx_map or j not in idx_map:
            continue
        pairs.append([idx_map[i], idx_map[j]])
        hi.append(float(bounds[i, j]))  # upper triangle = upper bound
        lo.append(float(bounds[j, i]))  # lower triangle = lower bound
        key = (i, j)
        isb.append(key in bond_set)
        isa.append(key in angle_set)
    if not pairs:
        return None
    return (
        np.asarray(pairs, np.int64).T,
        np.asarray(lo, np.float32),
        np.asarray(hi, np.float32),
        np.asarray(isb, bool),
        np.asarray(isa, bool),
    )


def _chiral_constraints(mol, idx_map):
    """SP3 chiral centres -> (quads[4,C], is_r[C]); 1 reference + 3 alternates."""
    import rdkit.Chem as Chem
    from rdkit.Chem.rdchem import HybridizationType

    if not all(a.HasProp("_CIPRank") for a in mol.GetAtoms()):
        return None
    quads, is_r = [], []
    for center_idx, orient in Chem.FindMolChiralCenters(mol, includeUnassigned=False):
        center = mol.GetAtomWithIdx(int(center_idx))
        if center.GetHybridization() != HybridizationType.SP3:
            continue
        nbrs = sorted(
            ((n.GetIdx(), int(n.GetProp("_CIPRank"))) for n in center.GetNeighbors()),
            key=lambda x: x[1], reverse=True,
        )
        nbr = tuple(n[0] for n in nbrs)
        if not 3 <= len(nbr) <= 4:
            continue
        r = orient == "R"
        quad = (*nbr[:3], int(center_idx))
        if all(i in idx_map for i in quad):
            quads.append([idx_map[i] for i in quad])
            is_r.append(r)
        if len(nbr) == 4:
            for skip in range(3):
                rest = nbr[:skip] + nbr[skip + 1:]
                quad = (rest[::-1] if skip % 2 == 0 else rest) + (int(center_idx),)
                if all(i in idx_map for i in quad):
                    quads.append([idx_map[i] for i in quad])
                    is_r.append(r)
    if not quads:
        return None
    return np.asarray(quads, np.int64).T, np.asarray(is_r, bool)


def _stereo_constraints(mol, idx_map):
    """E/Z bonds -> (quads[4,S], is_e[S]); primary + optional alternate."""
    import rdkit.Chem as Chem
    from rdkit.Chem.rdchem import BondStereo

    if not all(a.HasProp("_CIPRank") for a in mol.GetAtoms()):
        return None
    quads, is_e = [], []
    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if stereo not in (BondStereo.STEREOE, BondStereo.STEREOZ):
            continue
        sa, ea = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        sn = sorted(
            ((n.GetIdx(), int(n.GetProp("_CIPRank")))
             for n in mol.GetAtomWithIdx(sa).GetNeighbors() if n.GetIdx() != ea),
            key=lambda n: n[1], reverse=True,
        )
        en = sorted(
            ((n.GetIdx(), int(n.GetProp("_CIPRank")))
             for n in mol.GetAtomWithIdx(ea).GetNeighbors() if n.GetIdx() != sa),
            key=lambda n: n[1], reverse=True,
        )
        if not sn or not en:
            continue
        e = stereo == BondStereo.STEREOE
        quad = (sn[0][0], sa, ea, en[0][0])
        if all(i in idx_map for i in quad):
            quads.append([idx_map[i] for i in quad])
            is_e.append(e)
        if len(sn) == 2 and len(en) == 2:
            quad = (sn[1][0], sa, ea, en[1][0])
            if all(i in idx_map for i in quad):
                quads.append([idx_map[i] for i in quad])
                is_e.append(e)
    if not quads:
        return None
    return np.asarray(quads, np.int64).T, np.asarray(is_e, bool)


def _planar_constraints(mol, idx_map):
    """sp2-sp2 C=C planar groups -> improper quads[4, 2P]."""
    import rdkit.Chem as Chem

    smarts = Chem.MolFromSmarts(_PLANAR_SMARTS)
    quads = []
    for m in mol.GetSubstructMatches(smarts):
        if not all(i in idx_map for i in m):
            continue
        f = [idx_map[i] for i in m]  # 6 atoms: a0..a5
        # Two improper dihedrals keep the double bond planar (PlanarBondPotential).
        quads.append([f[1], f[2], f[3], f[0]])
        quads.append([f[4], f[5], f[0], f[3]])
    if not quads:
        return None
    return np.asarray(quads, np.int64).T


# ----------------------------------------------------------------------------
# Buffer baking: convert raw constraints to flat-bottom (lower, upper) bounds
# ----------------------------------------------------------------------------


def _bake_posebusters(pairs, lo, hi, isb, isa, atom_vdw):
    lower = lo.copy()
    upper = hi.copy()
    bb, ab, cb = _BOND_BUFFER, _ANGLE_BUFFER, _CLASH_BUFFER

    m = isb & ~isa
    lower[m] *= 1.0 - bb; upper[m] *= 1.0 + bb
    m = ~isb & isa
    lower[m] *= 1.0 - ab; upper[m] *= 1.0 + ab
    m = isb & isa
    lower[m] *= 1.0 - min(bb, ab); upper[m] *= 1.0 + min(bb, ab)
    m = ~isb & ~isa
    lower[m] *= 1.0 - cb; upper[m] = np.inf

    # VDW clash floor on non-bonded pairs; cap bonded uppers at the same cutoff.
    bond_cutoffs = 0.35 + atom_vdw[pairs].mean(axis=0)
    lower[~isb] = np.maximum(lower[~isb], bond_cutoffs[~isb])
    upper[isb] = np.minimum(upper[isb], bond_cutoffs[isb])
    return lower.astype(np.float32), upper.astype(np.float32)


def _bake_chiral(is_r):
    lower = np.where(is_r, _CHIRAL_BUFFER, -np.inf).astype(np.float32)
    upper = np.where(is_r, np.inf, -_CHIRAL_BUFFER).astype(np.float32)
    return lower, upper


def _bake_stereo(is_e):
    lower = np.where(is_e, math.pi - _STEREO_BUFFER, -np.inf).astype(np.float32)
    upper = np.where(is_e, np.inf, _STEREO_BUFFER).astype(np.float32)
    return lower, upper


def _bake_planar(n):
    lower = np.full(n, -np.inf, np.float32)
    upper = np.full(n, _PLANAR_BUFFER, np.float32)
    return lower, upper


# ----------------------------------------------------------------------------
# Inter-chain potentials: VDW overlap, connections, symmetric-chain COM.
# A full upper-triangular enumeration over real atoms then masking would be
# huge; here we enumerate only the kept (inter-chain, single-ion, non-connected)
# pairs directly so the JAX graph stays small.
# ----------------------------------------------------------------------------


def _per_atom_asym(asym_id_token, A):
    """[T] token asym ids -> [T*A] per-atom asym ids (each slot inherits token)."""
    return np.repeat(asym_id_token, A)


def _build_connections(example, A):
    """Inter-residue covalent bonds -> (atom_pairs[2,K], chain_pairs[2,K]).

    Every covalent bond that
    crosses a residue boundary (polymer-ligand, ligand-ligand inter-residue)
    becomes a soft-bond constraint, and the chains it links are recorded so
    VDW overlap skips them. AF3 exposes these as gather tables; same-residue
    (intra-ligand) bonds are already covered by the PoseBusters bounds and are
    excluded via ``ref_space_uid``.
    """
    asym_atom = None
    pdam = np.asarray(example["pred_dense_atom_mask"]).astype(bool)  # [T, A]
    ref_uid = np.asarray(example.get("ref_space_uid")) if "ref_space_uid" in example else None
    asym_tok = np.asarray(example["asym_id"])
    pairs, chain_pairs, seen = [], [], set()

    for key in ("token_atoms_to_polymer_ligand_bonds", "token_atoms_to_ligand_ligand_bonds"):
        gi = example.get(f"{key}:gather_idxs")
        gm = example.get(f"{key}:gather_mask")
        if gi is None or gm is None:
            continue
        gi = np.asarray(gi)
        gm = np.asarray(gm).astype(bool)
        # Each row that is fully masked-in encodes one bond's two flat atom
        # endpoints (token*A + slot). Layout: [N, 2] of flat atom indices.
        if gi.ndim != 2 or gi.shape[1] != 2:
            continue
        valid = gm.all(axis=-1)
        for flat_a, flat_b in gi[valid]:
            fa, fb = int(flat_a), int(flat_b)
            if fa < 0 or fb < 0:
                continue
            ta, tb = fa // A, fb // A
            if ta >= asym_tok.shape[0] or tb >= asym_tok.shape[0]:
                continue
            # Skip intra-residue bonds (already in PoseBusters bounds).
            if ref_uid is not None:
                ua = ref_uid[ta, fa % A]
                ub = ref_uid[tb, fb % A]
                if ua == ub and ua >= 0:
                    continue
            pairs.append([fa, fb])
            ca, cb = int(asym_tok[ta]), int(asym_tok[tb])
            if ca != cb:
                k = tuple(sorted((ca, cb)))
                if k not in seen:
                    seen.add(k)
                    chain_pairs.append(list(k))
    if not pairs:
        return (np.empty((2, 0), np.int64), np.empty((2, 0), np.int64))
    return (
        np.asarray(pairs, np.int64).T,
        np.asarray(chain_pairs, np.int64).T if chain_pairs else np.empty((2, 0), np.int64),
    )


def _build_vdw(asym_id_token, pdam, atom_vdw, A, connected_chain_pairs):
    """Inter-chain VDW-overlap pairs -> (index[2,P], lower[P]).

    Only real atoms, only chains with more than one atom (drop lone ions), and
    only chain pairs that are not
    covalently connected, lower bound = sum of VDW radii * (1 - buffer).
    """
    asym_atom = _per_atom_asym(asym_id_token, A)            # [T*A]
    atom_mask = pdam.reshape(-1).astype(bool)               # [T*A]
    real = np.nonzero(atom_mask)[0]
    if real.size < 2:
        return np.empty((2, 0), np.int64), np.empty((0,), np.float32)
    asym_real = asym_atom[real]
    # chain sizes over real atoms; keep only multi-atom chains (drop lone ions).
    chains, counts = np.unique(asym_real, return_counts=True)
    multi = set(int(c) for c, n in zip(chains, counts) if n > 1)
    connected = set()
    if connected_chain_pairs is not None and connected_chain_pairs.size:
        for a, b in connected_chain_pairs.T:
            connected.add(tuple(sorted((int(a), int(b)))))
    # group real-atom flat indices by chain
    by_chain = {int(c): real[asym_real == c] for c in chains if int(c) in multi}
    keep_chains = sorted(by_chain)
    pair_a, pair_b = [], []
    for i in range(len(keep_chains)):
        for j in range(i + 1, len(keep_chains)):
            ci, cj = keep_chains[i], keep_chains[j]
            if tuple(sorted((ci, cj))) in connected:
                continue
            ai = by_chain[ci]
            bj = by_chain[cj]
            aa, bb = np.meshgrid(ai, bj, indexing="ij")
            pair_a.append(aa.reshape(-1))
            pair_b.append(bb.reshape(-1))
    if not pair_a:
        return np.empty((2, 0), np.int64), np.empty((0,), np.float32)
    pa = np.concatenate(pair_a)
    pb = np.concatenate(pair_b)
    lower = (atom_vdw[pa] + atom_vdw[pb]) * (1.0 - _VDW_BUFFER)
    index = np.stack([pa, pb], axis=0).astype(np.int64)
    return index, lower.astype(np.float32)


def _build_symmetric_chains(example, pdam, A):
    """Symmetric (same-entity, multi-atom) chains -> (pairs[2,M], atom_chain[T*A], C).

    Symmetric-chain COM repulsion: chains sharing an ``entity_id`` repel
    each other's centre of mass. Returns ``pairs`` as *dense* chain-index pairs
    (0..C-1), a per-atom dense chain label (-1 for atoms not in any symmetric
    multi-atom chain), and the chain count C, so the JAX side can build per-chain
    COMs with a segment reduction. Empty for monomeric / lone-ion ligands.
    """
    asym_tok = np.asarray(example["asym_id"])
    ent_tok = np.asarray(example["entity_id"])
    asym_atom = _per_atom_asym(asym_tok, A)
    atom_mask = pdam.reshape(-1).astype(bool)
    real = np.nonzero(atom_mask)[0]
    asym_real = asym_atom[real]
    chains, counts = np.unique(asym_real, return_counts=True)
    multi = {int(c) for c, n in zip(chains, counts) if n > 1}
    asym_to_ent = {}
    for a, e in zip(asym_tok.tolist(), ent_tok.tolist()):
        asym_to_ent.setdefault(int(a), int(e))
    ent_to_asyms = {}
    for a in sorted(set(asym_tok.tolist())):
        if int(a) in multi:
            ent_to_asyms.setdefault(asym_to_ent[int(a)], []).append(int(a))
    pairs = []
    for grp in ent_to_asyms.values():
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                pairs.append([grp[i], grp[j]])
    if not pairs:
        return np.empty((2, 0), np.int64), np.empty((0, asym_atom.shape[0]), np.float32)
    # Dense-index the symmetric chains and build per-chain COM averaging weights
    # (row c sums to 1 over chain c's real atoms), so JAX gets COM = weight @ coords.
    used = sorted({c for p in pairs for c in p})
    dense = {c: i for i, c in enumerate(used)}
    N = asym_atom.shape[0]
    weight = np.zeros((len(used), N), np.float32)
    for c in used:
        members = (asym_atom == c) & atom_mask
        n = int(members.sum())
        if n > 0:
            weight[dense[c], members] = 1.0 / n
    dense_pairs = np.asarray([[dense[a], dense[b]] for a, b in pairs], np.int64).T
    return dense_pairs, weight


# ----------------------------------------------------------------------------
# Top-level builder
# ----------------------------------------------------------------------------


def build_steering_features(
    example: dict,
    ccd,
    *,
    ligand_only: bool = True,
    atoms_per_token: int = ATOMS_PER_TOKEN,
) -> Optional[Dict[str, np.ndarray]]:
    """Build grouped steering arrays from a featurised example dict.

    Args:
      example: a single featurised ``features.BatchDict`` (numpy arrays).
      ccd: an ``alphafold3.constants.chemical_components.Ccd`` instance.
      ligand_only: restrict constraints to ligand residues (cheap; the
        small-molecule validity signal posebuster scores). Set False to also
        constrain polymer residues (Boltz behaviour).

    Returns:
      dict of numpy arrays per group (``<group>_index/_lower/_upper``) or None
      if nothing constrainable was found.
    """
    layout = example["token_atoms_layout"]
    layout = layout.item() if isinstance(layout, np.ndarray) and layout.ndim == 0 else layout
    is_ligand = np.asarray(example["is_ligand"]).astype(bool)
    ref_element = np.asarray(example["ref_element"])  # [T, A] atomic numbers
    A = atoms_per_token
    n_tok = layout.shape[0]

    atom_vdw = _vdw_radii_table()[np.clip(ref_element, 0, NUM_ELEMENTS - 1)].reshape(-1)

    # Group tokens by residue (chain_id, res_id, res_name).
    groups: Dict[tuple, list] = {}
    for t in range(n_tok):
        res_name = ""
        for a in range(A):
            if layout.res_name[t, a]:
                res_name = layout.res_name[t, a]
                break
        if not res_name:
            continue
        if ligand_only and not is_ligand[t]:
            continue
        key = (layout.chain_id[t, 0], int(layout.res_id[t, 0]), res_name)
        groups.setdefault(key, []).append(t)

    pb_pairs, pb_lo, pb_hi = [], [], []
    ch_q, ch_lo, ch_hi = [], [], []
    st_q, st_lo, st_hi = [], [], []
    pl_q = []

    for (_chain, _resid, res_name), tokens in groups.items():
        tokens = sorted(tokens)
        mol = _ccd_mol(ccd, res_name)
        if mol is None:
            continue
        name_to_flat = {}
        for t in tokens:
            for a in range(A):
                nm = layout.atom_name[t, a]
                if nm:
                    name_to_flat[nm.strip()] = t * A + a
        idx_map = _idx_map(mol, name_to_flat)
        if not idx_map:
            continue

        geom = _geometry_constraints(mol, idx_map)
        if geom is not None:
            pairs, lo, hi, isb, isa = geom
            lower, upper = _bake_posebusters(pairs, lo, hi, isb, isa, atom_vdw)
            pb_pairs.append(pairs); pb_lo.append(lower); pb_hi.append(upper)

        chir = _chiral_constraints(mol, idx_map)
        if chir is not None:
            quads, is_r = chir
            lo_, hi_ = _bake_chiral(is_r)
            ch_q.append(quads); ch_lo.append(lo_); ch_hi.append(hi_)

        ster = _stereo_constraints(mol, idx_map)
        if ster is not None:
            quads, is_e = ster
            lo_, hi_ = _bake_stereo(is_e)
            st_q.append(quads); st_lo.append(lo_); st_hi.append(hi_)

        planar = _planar_constraints(mol, idx_map)
        if planar is not None:
            pl_q.append(planar)

    out: Dict[str, np.ndarray] = {}

    def _emit(name, idx_list, lo_list, hi_list, idx_rows):
        if not idx_list:
            return
        out[f"{name}_index"] = np.concatenate(idx_list, axis=1).astype(np.int32)
        out[f"{name}_lower"] = np.concatenate(lo_list).astype(np.float32)
        out[f"{name}_upper"] = np.concatenate(hi_list).astype(np.float32)

    _emit("posebusters", pb_pairs, pb_lo, pb_hi, 2)
    _emit("chiral", ch_q, ch_lo, ch_hi, 4)
    _emit("stereo", st_q, st_lo, st_hi, 4)
    if pl_q:
        pl = np.concatenate(pl_q, axis=1).astype(np.int32)
        lo_, hi_ = _bake_planar(pl.shape[1])
        out["planar_index"] = pl
        out["planar_lower"] = lo_
        out["planar_upper"] = hi_

    # --- Inter-chain potentials (VDW overlap, connections, symmetric-chain COM).
    pdam = np.asarray(example["pred_dense_atom_mask"]).astype(bool)
    asym_token = np.asarray(example["asym_id"])

    conn_index, conn_chains = _build_connections(example, A)
    if conn_index.shape[1]:
        out["connections_index"] = conn_index.astype(np.int32)
        out["connections_lower"] = np.full(conn_index.shape[1], -np.inf, np.float32)
        out["connections_upper"] = np.full(conn_index.shape[1], _CONNECTIONS_BUFFER, np.float32)

    vdw_index, vdw_lower = _build_vdw(asym_token, pdam, atom_vdw, A, conn_chains)
    if vdw_index.shape[1]:
        out["vdw_index"] = vdw_index.astype(np.int32)
        out["vdw_lower"] = vdw_lower
        out["vdw_upper"] = np.full(vdw_index.shape[1], np.inf, np.float32)

    sym_pairs, sym_weight = _build_symmetric_chains(example, pdam, A)
    if sym_pairs.shape[1]:
        # COM potential: buffer is time-scheduled so bounds are applied in JAX.
        out["symcom_pairs"] = sym_pairs.astype(np.int32)
        out["symcom_weight"] = sym_weight.astype(np.float32)

    return out or None
