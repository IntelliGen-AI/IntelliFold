set -e

# if HF_ENDPOINT is not set, use huggingface.co
if [ -z "$HF_ENDPOINT" ]; then
    HF_ENDPOINT="huggingface.co"
fi

# if cache_dir is not set, use "./cache_dir" by default
if [ -z "$cache_dir" ]; then
    cache_dir="./cache_data"
fi
mkdir -p "$cache_dir"

# if aria2 exists, use aria2 to download
if command -v aria2c &> /dev/null; then
    aria2c -c -x 10 -s 10 -k 1M -o cache_data/intellifold_v0.1.0.pt https://${HF_ENDPOINT}/intelligenAI/intellifold/resolve/main/intellifold_v0.1.0.pt
    aria2c -c -x 10 -s 10 -k 1M -o cache_data/intellifold_v2.pt https://${HF_ENDPOINT}/intelligenAI/intellifold/resolve/main/intellifold_v2.pt
    aria2c -c -x 10 -s 10 -k 1M -o cache_data/intellifold_v2_flash.pt https://${HF_ENDPOINT}/intelligenAI/intellifold/resolve/main/intellifold_v2_flash.pt
    aria2c -c -x 10 -s 10 -k 1M -o cache_data/ccd.pkl https://${HF_ENDPOINT}/intelligenAI/intellifold/resolve/main/ccd.pkl
    aria2c -c -x 10 -s 10 -k 1M -o cache_data/unique_protein_sequences.fasta https://${HF_ENDPOINT}/intelligenAI/intellifold/resolve/main/unique_protein_sequences.fasta
    aria2c -c -x 10 -s 10 -k 1M -o cache_data/unique_nucleic_acid_sequences.fasta https://${HF_ENDPOINT}/intelligenAI/intellifold/resolve/main/unique_nucleic_acid_sequences.fasta
    aria2c -c -x 10 -s 10 -k 1M -o cache_data/protein_id_groups.json https://${HF_ENDPOINT}/intelligenAI/intellifold/resolve/main/protein_id_groups.json
    aria2c -c -x 10 -s 10 -k 1M -o cache_data/nucleic_acid_id_groups.json https://${HF_ENDPOINT}/intelligenAI/intellifold/resolve/main/nucleic_acid_id_groups.json

else
    wget -P ./cache_data https://${HF_ENDPOINT}/intelligenAI/intellifold/resolve/main/intellifold_v0.1.0.pt
    wget -P ./cache_data https://${HF_ENDPOINT}/intelligenAI/intellifold/resolve/main/intellifold_v2.pt
    wget -P ./cache_data https://${HF_ENDPOINT}/intelligenAI/intellifold/resolve/main/intellifold_v2_flash.pt
    wget -P ./cache_data https://${HF_ENDPOINT}/intelligenAI/intellifold/resolve/main/ccd.pkl
    wget -P ./cache_data https://${HF_ENDPOINT}/intelligenAI/intellifold/resolve/main/unique_protein_sequences.fasta
    wget -P ./cache_data https://${HF_ENDPOINT}/intelligenAI/intellifold/resolve/main/unique_nucleic_acid_sequences.fasta
    wget -P ./cache_data https://${HF_ENDPOINT}/intelligenAI/intellifold/resolve/main/protein_id_groups.json
    wget -P ./cache_data https://${HF_ENDPOINT}/intelligenAI/intellifold/resolve/main/nucleic_acid_id_groups.json
fi

