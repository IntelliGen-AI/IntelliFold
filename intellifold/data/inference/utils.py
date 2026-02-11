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

import os
import urllib.request
from pathlib import Path

### Add Request Header
opener = urllib.request.build_opener()
opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "huggingface.co")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#### huggingface offical URL
CCD_URL = f"https://{HF_ENDPOINT}/intelligenAI/intellifold/resolve/main/ccd.pkl"
MODEL_URL = (
    f"https://{HF_ENDPOINT}/intelligenAI/intellifold/resolve/main/intellifold_v0.1.0.pt"
)
V2_FLASH_MODEL_URL = (
    f"https://{HF_ENDPOINT}/intelligenAI/intellifold/resolve/main/intellifold_v2_flash.pt"
)
V2_MODEL_URL = (
    f"https://{HF_ENDPOINT}/intelligenAI/intellifold/resolve/main/intellifold_v2.pt"
)
PROTEIN_PDB_SEQUENCES_URL = f"https://{HF_ENDPOINT}/intelligenAI/intellifold/resolve/main/unique_protein_sequences.fasta"
RNA_PDB_SEQUENCES_URL = f"https://{HF_ENDPOINT}/intelligenAI/intellifold/resolve/main/unique_nucleic_acid_sequences.fasta"
PROTEIN_PDB_GROUPS_URL = f"https://{HF_ENDPOINT}/intelligenAI/intintellifoldfold/resolve/main/protein_id_groups.json"
RNA_PDB_GROUPS_URL = f"https://{HF_ENDPOINT}/intelligenAI/intellifold/resolve/main/nucleic_acid_id_groups.json"


#### huggingface-mirror URL
CCD_MIRROR_URL = "https://hf-mirror.com/intelligenAI/intellifold/resolve/main/ccd.pkl"
MODEL_MIRROR_URL = (
    "https://hf-mirror.com/intelligenAI/intellifold/resolve/main/intellifold_v0.1.0.pt"
)
V2_FLASH_MODEL_MIRROR_URL = (
    "https://hf-mirror.com/intelligenAI/intellifold/resolve/main/intellifold_v2_flash.pt"
)
V2_MODEL_MIRROR_URL = (
    "https://hf-mirror.com/intelligenAI/intellifold/resolve/main/intellifold_v2.pt"
)
PROTEIN_PDB_SEQUENCES_MIRROR_URL = "https://hf-mirror.com/intelligenAI/intellifold/resolve/main/unique_protein_sequences.fasta"
RNA_PDB_SEQUENCES_MIRROR_URL = "https://hf-mirror.com/intelligenAI/intellifold/resolve/main/unique_nucleic_acid_sequences.fasta"
PROTEIN_PDB_GROUPS_MIRROR_URL = "https://hf-mirror.com/intelligenAI/intellifold/resolve/main/protein_id_groups.json"
RNA_PDB_GROUPS_MIRROR_URL = "https://hf-mirror.com/intelligenAI/intellifold/resolve/main/nucleic_acid_id_groups.json"

    
def download(cache: Path, model: str) -> None:
    """Download all the required data.

    Parameters
    ----------
    cache : Path
        The cache directory.

    """
    # Download CCD
    ccd = cache / "ccd.pkl"
    if not ccd.exists():
        print(
            f"Downloading the CCD dictionary to {ccd}. You may "
            "change the cache directory with the --cache flag."
        )
        try:
            urllib.request.urlretrieve(CCD_URL, str(ccd)) 
        except:
            ### use hf-mirror.com
            urllib.request.urlretrieve(CCD_MIRROR_URL, str(ccd))
            
    # Download model
    if model == "v1":
        model = cache / "intellifold_v0.1.0.pt"
        if not model.exists():
            print(
                f"Downloading the model weights to {model}. You may "
                "change the cache directory with the --cache flag."
            )
            try:
                urllib.request.urlretrieve(MODEL_URL, str(model))  
            except:
                urllib.request.urlretrieve(MODEL_MIRROR_URL, str(model))
                
    elif model == "v2-flash":
        # Download v2 flash model
        v2_flash_model = cache / "intellifold_v2_flash.pt"
        if not v2_flash_model.exists():
            print(
                f"Downloading the v2 flash model weights to {v2_flash_model}. You may "
                "change the cache directory with the --cache flag."
            )
            try:
                urllib.request.urlretrieve(V2_FLASH_MODEL_URL, str(v2_flash_model))  
            except:
                urllib.request.urlretrieve(V2_FLASH_MODEL_MIRROR_URL, str(v2_flash_model))
    elif model == "v2":
        # Download v2 model
        v2_model = cache / "intellifold_v2.pt"
        if not v2_model.exists():
            print(
                f"Downloading the v2 model weights to {v2_model}. You may "
                "change the cache directory with the --cache flag."
            )
            try:
                urllib.request.urlretrieve(V2_MODEL_URL, str(v2_model))  
            except:
                urllib.request.urlretrieve(V2_MODEL_MIRROR_URL, str(v2_model))
    else:
        raise ValueError(f"Invalid model: {model} for downloading model weights.")
            
    # Download Protein Sequences database
    protein_sequences = cache / "unique_protein_sequences.fasta"
    if not protein_sequences.exists():
        print(
            f"Downloading the protein sequences to {protein_sequences}. You may "
            "change the cache directory with the --cache flag."
        )
        try:
            urllib.request.urlretrieve(PROTEIN_PDB_SEQUENCES_URL, str(protein_sequences))  
        except:
            urllib.request.urlretrieve(PROTEIN_PDB_SEQUENCES_MIRROR_URL, str(protein_sequences))
    # Download RNA Sequences database
    rna_sequences = cache / "unique_nucleic_acid_sequences.fasta"
    if not rna_sequences.exists():
        print(
            f"Downloading the RNA sequences to {rna_sequences}. You may "
            "change the cache directory with the --cache flag."
        )
        try:
            urllib.request.urlretrieve(RNA_PDB_SEQUENCES_URL, str(rna_sequences))  
        except:
            urllib.request.urlretrieve(RNA_PDB_SEQUENCES_MIRROR_URL, str(rna_sequences))
    # Download protein id groups
    protein_groups = cache / "protein_id_groups.json"
    if not protein_groups.exists():
        print(
            f"Downloading the protein id groups to {protein_groups}. You may "
            "change the cache directory with the --cache flag."
        )
        try:
            urllib.request.urlretrieve(PROTEIN_PDB_GROUPS_URL, str(protein_groups))  
        except:
            urllib.request.urlretrieve(PROTEIN_PDB_GROUPS_MIRROR_URL, str(protein_groups))
    # Download RNA id groups
    rna_groups = cache / "nucleic_acid_id_groups.json"
    if not rna_groups.exists():
        print(
            f"Downloading the RNA id groups to {rna_groups}. You may "
            "change the cache directory with the --cache flag."
        )
        try:
            urllib.request.urlretrieve(RNA_PDB_GROUPS_URL, str(rna_groups))  
        except:
            urllib.request.urlretrieve(RNA_PDB_GROUPS_MIRROR_URL, str(rna_groups))
            


def get_cache_path() -> str:
    """Determine the cache path, prioritising the INTELLIFOLD_CACHE environment variable.

    Returns
    -------
    str: Path
        Path to use for intellifold cache location.

    """
    env_cache = os.environ.get("INTELLIFOLD_CACHE")
    if env_cache:
        resolved_cache = Path(env_cache).expanduser().resolve()
        if not resolved_cache.is_absolute():
            raise ValueError(f"INTELLIFOLD_CACHE must be an absolute path, got: {env_cache}")
        return str(resolved_cache)

    return str(Path("~/.intellifold").expanduser())