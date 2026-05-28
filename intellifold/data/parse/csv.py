# Copyright (c) 2024 Jeremy Wohlwend, Gabriele Corso, Saro Passaro
#
# Licensed under the MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import string
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from intellifold.data import const
from intellifold.data.types import MSA, MSADeletion, MSAResidue, MSASequence


def parse_csv(
    path: Path,
    max_seqs: Optional[int] = None,
) -> MSA:
    """Process an A3M file.

    Parameters
    ----------
    path : Path
        The path to the a3m(.gz) file.
    max_seqs : int, optional
        The maximum number of sequences.

    Returns
    -------
    MSA
        The MSA object.

    """
    # Read file
    data = pd.read_csv(path)

    # Check columns
    if tuple(sorted(data.columns)) != ("key", "sequence"):
        msg = "Invalid CSV format, expected columns: ['sequence', 'key']"
        raise ValueError(msg)

    # Create taxonomy mapping
    # Dedup SEPARATELY for paired (key>=0) and unpaired (key==-1) sequences.
    # PV deduplicates each A3M independently; paired and unpaired are separate files.
    # Cross-dedup happens later in PV (after profile/deletion_mean are computed).
    _deletion_table = str.maketrans("", "", string.ascii_lowercase)
    visited_paired = set()
    visited_unpaired = set()
    sequences = []
    deletions = []
    residues = []

    seq_idx = 0
    for line, key in zip(data["sequence"], data["key"]):
        line: str
        line = line.strip()  # noqa: PLW2901
        if not line:
            continue

        # Get taxonomy, if annotated
        taxonomy_id = -1
        if (str(key) != "nan") and (key is not None) and (key != ""):
            taxonomy_id = key

        # Skip if duplicate sequence within same group (paired vs unpaired).
        # Matching PV dedup: remove lowercase insertions, keep uppercase + gaps.
        str_seq = line.translate(_deletion_table)
        is_unpaired = (taxonomy_id == -1)
        visited_set = visited_unpaired if is_unpaired else visited_paired
        if str_seq not in visited_set:
            visited_set.add(str_seq)
        else:
            continue

        # Process sequence
        residue = []
        deletion = []
        count = 0
        res_idx = 0
        for c in line:
            if c != "-" and c.islower():
                count += 1
                continue
            token = const.prot_letter_to_token[c]
            # token = const.token_ids[token]
            token=const.mapping_boltz_token_ids_to_our_token_ids[const.token_ids[token]]
            residue.append(token)
            if count > 0:
                deletion.append((res_idx, count))
                count = 0
            res_idx += 1

        res_start = len(residues)
        res_end = res_start + len(residue)

        del_start = len(deletions)
        del_end = del_start + len(deletion)

        sequences.append((seq_idx, taxonomy_id, res_start, res_end, del_start, del_end))
        residues.extend(residue)
        deletions.extend(deletion)

        seq_idx += 1
        if (max_seqs is not None) and (seq_idx >= max_seqs):
            break
    # Create MSA object
    msa = MSA(
        residues=np.array(residues, dtype=MSAResidue),
        deletions=np.array(deletions, dtype=MSADeletion),
        sequences=np.array(sequences, dtype=MSASequence),
    )
    return msa
