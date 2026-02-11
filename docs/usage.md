### Run with Command line

If you set up `IntelliFold` by `pip`, you can run the following command to do model inference:

```bash

# run with example yaml, which contains precomputed msa files(.a3m or .csv). the default seed is 42.
## a3m MSA file type
intellifold predict ./examples/5S8I_A.yaml --out_dir ./output 
## csv MSA file type
intellifold predict ./examples/7yds.yaml --out_dir ./output 

# Predict with a directory of YAMLs
intellifold predict ./examples --out_dir ./output

# run with 5 seeds(mutiple seeds are splited them by comma) and 5 samples (the default parameters for AlphaFold3).
intellifold predict ./examples/5S8I_A.yaml --seed 42,43,44,45,46 --num_diffusion_samples 5 --out_dir ./output

# if the input yaml file do not contain precomputed msa paths, you can set --use_msa_server to search msa(need internet connection and would take some time) 
# and use greedy msa pairing strategy
# and then predict
intellifold predict examples/examples_wo_msa/example_without_msa.yaml --out_dir ./output --seed 42,66 --use_msa_server --msa_pairing_strategy greedy

# only run the data processing step, and not run the model.
intellifold predict ./examples/5S8I_A.yaml --out_dir ./output --only_run_data_process

# choose the model version to run, the default is v2-flash, which is faster and more accurate than v1 and v2. you can choose the model based on your needs and computational resources.
intellifold predict ./examples/5S8I_A.yaml --out_dir ./output --model v2
```

## MSA Server Authentication

When using custom MSA servers that require authentication (such as enterprise or private deployments), IntelliFold supports both basic authentication and API key authentication.

### Quick Setup

**For API Key Authentication (recommended):**
```bash
# Set environment variable (secure)
export MSA_API_KEY_VALUE=your-api-key

# Run with custom MSA server
intellifold predict input.yaml --out_dir ./output --use_msa_server 
  --msa_server_url https://your-msa-server.com 
  --api_key_header X-API-Key
```

**For Basic Authentication:**
```bash
# Set environment variables (secure)
export MSA_USERNAME=your-username
export MSA_PASSWORD=your-password

# Run with custom MSA server
intellifold predict input.yaml --out_dir ./output --use_msa_server 
  --msa_server_url https://your-msa-server.com
```

### Authentication Options

- **Environment variables** (recommended for security): `MSA_API_KEY_VALUE`, `MSA_USERNAME`, `MSA_PASSWORD`
- **Custom headers**: Use `--api_key_header` for different API key header names (e.g., `X-Gravitee-Api-Key`)
- **Multiple auth types**: Cannot use both basic auth and API key simultaneously
- **Backward compatibility**: All authentication is optional - existing workflows continue to work

The public ColabFold server (`https://api.colabfold.com`) requires no authentication and remains the default.

### Run with Bash Script

The aurguments is the same as `intellifold predict`, and you can set the parameters in the script.
you can get the help information by running `intellifold predict --help` or `python run_intellifold.py --help`
```bash
bash predict.sh
## or
## python run_intellifold.py ....
```
Common arguments of this `scripts`/`intellifold predict` are explained as follows:
* `--out_dir` (`PATH`, default: `./`)  
  The path where to save the predictions.
* `--cache` (`PATH`, default: `~/.intellifold`)  
  The directory where to download the data and model. Will use environment variable `INTELLIFOLD_CACHE` as an absolute path if set.
* `--num_workers` (`INTEGER`, default: `4`)  
  The number of dataloader workers to use for prediction.
* `--precision` (`str`, default: `bf16`)  
  Sets precision, lower precision improves runtime.
* `--seed` (`INTEGER`, default: `42`)  
  Random seed (single int or multiple ints separated by comma, e.g., '42' or '42,43').
* `--recycling_iters` (`INTEGER`, default: `10`)  
  Number of recycling iterations.
* `--num_diffusion_samples` (`INTEGER`, default: `5`)  
  The number of diffusion samples.
* `--sampling_steps` (`INTEGER`, default: `200`)  
  The number of diffusion sampling steps to use.
* `--output_format` (`[pdb,mmcif]`, default: `mmcif`)  
  The output format to use for the predictions (pdb or mmcif).
* `--override` (`FLAG`, default: `False`)  
  Whether to override existing found predictions.
* `--use_msa_server` (`FLAG`, default: `False`)  
  Whether to use the MMSeqs2 server for MSA generation.
* `--msa_server_url` (`str`, default: `https://api.colabfold.com`)  
  MSA server url. Used only if `--use_msa_server` is set.
* `--msa_server_username` (`str`, default: `None`)  
  Username for basic authentication to MSA server. Can use environment variable `MSA_USERNAME`.
* `--msa_server_password` (`str`, default: `None`)  
  Password for basic authentication to MSA server. Can use environment variable `MSA_PASSWORD`.
* `--api_key_header` (`str`, default: `X-API-Key`)  
  Header name for API key authentication to MSA server.
* `--api_key_value` (`str`, default: `None`)  
  API key value for authentication to MSA server. Can use environment variable `MSA_API_KEY_VALUE`.
* `--msa_pairing_strategy` (`str`, default: `complete`)  
  Pairing strategy to use. Used only if `--use_msa_server` is set. Options are 'greedy' and 'complete'.
* `--no_pairing` (`FLAG`, default: `False`)  
  Whether to use pairing for Protein Multimer MSA generation.
* `--only_run_data_process` (`FLAG`, default: `False`)  
  Whether to only run data processing, and not run the model.
* `--return_similar_seq` (`FLAG`, default: `False`)
  Whether to return sequences similar to those in the training PDB dataset during inference. You can use these similar sequences and its PDB ids to do further analysis, such as a reference structure.
  > Before using this option, please make sure the mmseqs2 tool is installed, you can install it by running `conda install -c conda-forge -c bioconda mmseqs2`
* `--model` (`[v1, v2, v2-flash]`, default: `v2-flash`)  
  The model to use for prediction. Options are 'v1', 'v2', and 'v2-flash'. 'v2-flash' is the default and recommended model, which is faster and more accurate than 'v1' and 'v2'. 'v1' is the original model used in the IntelliFold paper, and 'v2' is an improved version of the model with better performance but slower inference speed than 'v2-flash'. You can choose the model based on your needs and computational resources.