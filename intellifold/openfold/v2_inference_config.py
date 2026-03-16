import re
import copy
import importlib
import ml_collections as mlc
import os
from intellifold.openfold.utils.kernel.deepspeed_compat import resolve_ds4s_request

def model_config(
    low_prec=False, 
    use_deepspeed_evoformer_attention=False,
):  
    
    c = copy.deepcopy(config)
    # TRAINING PRESETS
    if use_deepspeed_evoformer_attention:
        c.globals.use_deepspeed_evo_attention = True 
    if low_prec:
        c.globals.eps = 1e-4
        # c.globals.inf = 1e4

    return c

def get_model_config(args):
    """
    Returns the configuration for the IntelliFold model.
    """
    # Set the configuration parameters
    # Note: The following parameters are set to their default values.
    # You can modify them as per your requirements.
    
    is_low_precision = True
    if os.environ.get("USE_DEEPSPEED_EVO_ATTENTION", False) == "true":
        use_deepspeed_evoformer_attention = resolve_ds4s_request(True)
        cutlass_path_env = os.getenv("CUTLASS_PATH", None)
        msg = (
                "if use ds4sci, set `CUTLASS_PATH` environment variable according to the instructions at https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/. \n"
                "Or, you can refer the docs/kernels.md, you can set environment variable `CUTLASS_PATH` as follows: \n"
                "git clone -b v3.5.1 https://github.com/NVIDIA/cutlass.git  /path/to/cutlass \n"
                "export CUTLASS_PATH=/path/to/cutlass \n"
            )
        if use_deepspeed_evoformer_attention:
            assert (
                cutlass_path_env is not None and os.path.exists(cutlass_path_env)
            ), msg
    else:
        use_deepspeed_evoformer_attention = False
    
    config = model_config(
        low_prec=is_low_precision,
        use_deepspeed_evoformer_attention=use_deepspeed_evoformer_attention,
    )
    config.sample.no_sample_steps_T = args.sampling_steps
    config.backbone.recycling_iters = args.recycling_iters

    # Enable v2 inference features
    config.globals.advanced_conversion = True
    config.diffusion.atom_attention_encoder.advanced_conversion = True
    config.diffusion.atom_attention_decoder.advanced_conversion = True
    config.backbone.input_embedder.advanced_conversion = True
    config.backbone.msa.msa_stack.skip_unused_modules = True

    return config


c_z = mlc.FieldReference(512, field_type=int) # original 128
no_heads_pair = mlc.FieldReference(8, field_type=int) # original 4
c_m = mlc.FieldReference(256, field_type=int) # original 64
no_heads_msa = mlc.FieldReference(8, field_type=int) # original 8
c_t = mlc.FieldReference(256, field_type=int) # original 64
no_heads_template = mlc.FieldReference(8, field_type=int) # original 4

c_s = mlc.FieldReference(384, field_type=int)


c_s_inputs = mlc.FieldReference(384 + 31 + 31 + 1, field_type=int)
no_heads_atom = mlc.FieldReference(4, field_type=int)

c_atom = mlc.FieldReference(128, field_type=int)
c_atompair = mlc.FieldReference(16, field_type=int)
c_token = mlc.FieldReference(768, field_type=int) # 384 in the input embedder
sigma_data = mlc.FieldReference(16, field_type=float)
confidence_enabled = mlc.FieldReference(True, field_type=bool)

chunk_size = mlc.FieldReference(4, field_type=int)
aux_distogram_bins = mlc.FieldReference(64, field_type=int)
tm_enabled = mlc.FieldReference(False, field_type=bool)
eps = mlc.FieldReference(1e-8, field_type=float)
inf = mlc.FieldReference(1e9, field_type=float)
templates_enabled = mlc.FieldReference(True, field_type=bool)
tune_chunk_size = mlc.FieldReference(True, field_type=bool)
advanced_conversion = mlc.FieldReference(False, field_type=bool)
sampling_steps = mlc.FieldReference(200, field_type=int)

config = mlc.ConfigDict(
    {
        "globals": {
            "chunk_size": chunk_size,
            # Use DeepSpeed memory-efficient attention kernel
            "use_deepspeed_evo_attention": False,
            "c_z": c_z,
            "no_heads_pair": no_heads_pair,
            "c_m": c_m,
            "no_heads_msa": no_heads_msa,
            "c_t": c_t,
            "no_heads_template": no_heads_template,
            "c_s": c_s,
            "c_s_inputs": c_s_inputs,
            "c_atom": c_atom,
            "no_heads_atom": no_heads_atom,
            "c_atompair": c_atompair,
            "c_token": c_token,
            "sigma_data": sigma_data,
            "confidence_enabled": confidence_enabled,
            "eps": eps,
            "inf": inf,
            "advanced_conversion": False,
        },
        
        "backbone": {
            "recycling_iters" : 3,
            "_mask_trans": True,
            "input_embedder": {
                "c_z": c_z,
                "c_s": c_s,
                "c_s_inputs": c_s_inputs,
                "c_atom": c_atom,
                "c_atompair": c_atompair,
                "c_token": c_s,
                "c_ref": 3 + 1 + 128 + 1 + 4 * 64,
                "no_blocks": 3,
                "no_heads": no_heads_atom,
                "window_size_row": 32,
                "window_size_col": 128,
                "r_max" :32,
                "s_max": 2,
                "inf": inf,
                "eps": eps,  # 1e-6
                "tune_chunk_size": tune_chunk_size,
                "advanced_conversion": False,
            },
            "recycling_embedder": {
                "c_s": c_s,
                "c_z": c_z,
                "eps": eps,  # 1e-6,
                "inf": inf,
            },
            "template_embedder": {
                "c_z": c_z,
                "c_a": 39 + 1 + 3 + 1 + 31 + 31,
                "no_bins": 39,
                "c_t": c_t,
                "no_blocks": 2,
                "c_hidden_mul": c_t,
                "c_hidden_pair_att": c_t // (no_heads_template),
                "no_heads_pair": no_heads_template,
                "transition_n": 2,
                "pair_dropout": 0.25,
                "tune_chunk_size": tune_chunk_size,
                "inf": inf,
                "eps": eps,  # 1e-6,
                "enabled": templates_enabled,
            },
            "msa": {
                "msa_embedder": {
                    "c_msa_feat": 34,
                    "c_m": c_m,
                    "c_s_inputs": c_s_inputs,
                    "msa_depth": 1024,
                },
                "msa_stack": {
                    "c_m": c_m,
                    "c_z": c_z,
                    "c_hidden_msa_att": c_m // no_heads_msa,
                    "c_hidden_opm": 32,
                    "c_hidden_mul": c_z,
                    "c_hidden_pair_att": c_z // no_heads_pair,
                    "no_heads_msa": no_heads_msa,
                    "no_heads_pair": no_heads_pair,
                    "no_blocks": 4,
                    "transition_n": 4,
                    "msa_dropout": 0.15,
                    "pair_dropout": 0.25,
                    "inf": inf,
                    "eps": eps,  # 1e-10,
                    "tune_chunk_size": tune_chunk_size,
                    "skip_unused_modules": False,
                },
            },
            "pairformer_stack": {
                "c_s": c_s,
                "c_z": c_z,
                "c_hidden_mul": c_z,
                "c_hidden_pair_att": c_z // no_heads_pair,
                "no_heads_single": 16,
                "no_heads_pair": no_heads_pair,
                "no_blocks": 48,
                "transition_n": 4,
                "pair_dropout": 0.25,
                "tune_chunk_size": tune_chunk_size,
                "inf": inf,
                "eps": eps,  # 1e-10,
            },
            "heads": {
                "distogram": {
                    "c_z": c_z,
                    "no_bins": aux_distogram_bins,
                },
                },
            },
        "diffusion": {
            "window_size_row": 32,
            "window_size_col": 128,
            "sigma_data": sigma_data,
            "diffusion_conditioning": {
                "c_z": c_z,
                "c_s": c_s,
                "c_s_inputs": c_s_inputs,
                "c_fourier": 256,
                "sigma_data": sigma_data,
                "no_transitions":2 ,
                "transition_n": 2,
                "r_max": 32,
                "s_max": 2,
                "inf": inf,
                "eps": eps,  # 1e-6,
                "advanced_conditioning": False,
            },
            "atom_attention_encoder": {
                "c_atom": c_atom,
                "c_atompair": c_atompair,
                "c_token": c_token,
                "c_s": c_s,
                "c_z": c_z,
                "c_ref": 3 + 1 + 128 + 1 + 4 * 64,
                "no_blocks": 3,
                "no_heads": 4,
                "window_size_row": 32,
                "window_size_col": 128,
                "tune_chunk_size": tune_chunk_size,
                "inf": inf,
                "eps": eps,  # 1e-10,
                "advanced_conversion": False,
            },
            "diffusion_transformer": {
                "c_a": c_token,
                "c_z": c_z,
                "c_s": c_s,
                "no_blocks": 24,
                "no_heads": 16,
                "transition_n": 2,
                "tune_chunk_size": tune_chunk_size,
                "inf": inf,
                "eps": eps,  # 1e-10,
            },
            "atom_attention_decoder": {
                "c_atom": c_atom,
                "c_atompair": c_atompair,
                "c_token": c_token,
                "no_blocks": 3,
                "no_heads": 4,
                "window_size_row": 32,
                "window_size_col": 128,
                "tune_chunk_size": tune_chunk_size,
                "inf": inf,
                "eps": eps,  # 1e-10,
                "advanced_conversion": False,
            },
        },
        "confidence_head": {
            "enabled": confidence_enabled,
            "_mask_trans": True,
            "c_z": c_z,
            "c_s": c_s,
            "c_s_inputs": c_s_inputs,
            "no_bin_pae": 64,
            "no_bin_pde": 64,
            "no_bin_plddt": 50,
            "min_bin": 3.25,
            "max_bin": 50.75,
            "no_bins": 39,
            "max_num_atoms" : 24,
            "eps": eps,  # 1e-6,
            "inf": inf,
            "pairformer_stack": {
                "c_s": c_s,
                "c_z": c_z,
                "c_hidden_mul": c_z,
                "c_hidden_pair_att": c_z // no_heads_pair,
                "no_heads_single": 16,
                "no_heads_pair": no_heads_pair,
                "no_blocks": 4,
                "transition_n": 4,
                "pair_dropout": 0.25,
                "tune_chunk_size": tune_chunk_size,
                "inf": inf,
                "eps": eps,  # 1e-10,
            },
        },
        "sample": {
            "sigma_data": sigma_data,
            "no_sample_steps_T": sampling_steps,
            "mini_roll_out_steps_T": 20,
            "sigma_max": 160,
            "sigma_min": 4e-4,
            "rho": 7,
            "P_mean": -1.2,
            "P_std":1.5,
            "gamma_0": 0.8,
            "gamma_min": 1.0,
            "noise_scale_lambda": 1.003,
            "step_scale_eta": 1.5,
        },
    }
)
