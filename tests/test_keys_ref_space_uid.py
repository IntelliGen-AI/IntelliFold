"""keys_ref_space_uid: queries-layout gather for IntelliFold-v2, AF3 behaviour otherwise.

Toy layout (3 tokens with 2, 3 and 1 atoms, 4 dense slots per token; 2 query
subsets of 4; every subset's keys are all 8 query slots). The same-reference-space
mask must mark exactly the query/key pairs whose atoms belong to the same token.
"""
import os

import numpy as np
import pytest

from alphafold3.model.atom_layout import atom_layout
from alphafold3.model.network import atom_cross_attention as aca

# token-atom layout (num_tokens=3, num_dense=4); uid = token index + 1, 0 = padding
UID_TOKEN_ATOMS = np.array([[1, 1, 0, 0],
                            [2, 2, 2, 0],
                            [3, 0, 0, 0]])
# queries layout packs the 6 real atoms: [a0 a1 b0 b1 | b2 c0 pad pad]
TOKEN_ATOMS_TO_QUERIES = atom_layout.GatherInfo(
    gather_idxs=np.array([[0, 1, 4, 5], [6, 8, 0, 0]]),
    gather_mask=np.array([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=bool),
    input_shape=np.array([3, 4]),
)
# keys index the flattened QUERIES layout (input_shape == queries shape)
QUERIES_TO_KEYS = atom_layout.GatherInfo(
    gather_idxs=np.tile(np.arange(8), (2, 1)),
    gather_mask=np.tile(np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=bool), (2, 1)),
    input_shape=np.array([2, 4]),
)
TRUE_KEY_UID = np.array([1, 1, 2, 2, 2, 3])          # uid of each real key atom


def _mask(from_queries):
  q_uid = atom_layout.convert(TOKEN_ATOMS_TO_QUERIES, UID_TOKEN_ATOMS, layout_axes=(-2, -1))
  k_uid = aca._keys_ref_space_uid(QUERIES_TO_KEYS, q_uid, UID_TOKEN_ATOMS, from_queries)
  return q_uid, k_uid, q_uid[:, :, None] == k_uid[:, None, :]


def test_queries_layout_gather_marks_exactly_same_conformer_pairs():
  q_uid, k_uid, valid = _mask(from_queries=True)
  np.testing.assert_array_equal(k_uid[:, :6], np.tile(TRUE_KEY_UID, (2, 1)))
  truth = q_uid[:, :, None] == TRUE_KEY_UID[None, None, :]
  real_q = TOKEN_ATOMS_TO_QUERIES.gather_mask
  np.testing.assert_array_equal(valid[:, :, :6][real_q], truth[real_q])


def test_af3_default_reads_other_atoms_uid():
  # Upstream behaviour (google-deepmind/alphafold3#730): key k reads token-atom
  # slot k, so from the second token on it is another atom's uid or padding.
  _, k_uid, _ = _mask(from_queries=False)
  np.testing.assert_array_equal(k_uid[0, :6], [1, 1, 0, 0, 2, 2])
  assert not np.array_equal(k_uid[0, :6], TRUE_KEY_UID)


def test_flag_defaults_off_and_full_fat_turns_it_on_for_both_encoders(monkeypatch):
  from alphafold3.model import model
  from intellifold import patches

  stock = model.Model.Config()
  assert stock.evoformer.per_atom_conditioning.keys_ref_space_uid_from_queries is False
  assert stock.heads.diffusion.keys_ref_space_uid_from_queries is False

  monkeypatch.setenv('INTFOLD_FULLFAT', '1')
  cfg = model.Model.Config()
  assert patches.widen_config_full_fat(cfg)
  assert cfg.evoformer.per_atom_conditioning.keys_ref_space_uid_from_queries is True
  assert cfg.heads.diffusion.keys_ref_space_uid_from_queries is True


def test_stock_af3_config_is_untouched_without_full_fat(monkeypatch):
  from alphafold3.model import model
  from intellifold import patches

  monkeypatch.delenv('INTFOLD_FULLFAT', raising=False)
  cfg = model.Model.Config()
  assert not patches.widen_config_full_fat(cfg)
  assert cfg.evoformer.per_atom_conditioning.keys_ref_space_uid_from_queries is False
  assert cfg.heads.diffusion.keys_ref_space_uid_from_queries is False
