
import os
from intellifold.openfold.config import model_config
from intellifold.openfold.utils.kernel.deepspeed_compat import resolve_ds4s_request
    
def get_model_config(args):
    """
    Returns the configuration for the IntelliFold model.
    """
    # Set the configuration parameters
    # Note: The following parameters are set to their default values.
    # You can modify them as per your requirements.
    
    # Model configuration
    # These parameters are used to configure the model architecture and training settings.
    
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
    
    # Update hyper-parameters for v2 flash model
    config.backbone.pairformer_stack.no_blocks = 12
    config.diffusion.atom_attention_encoder.no_blocks = 2
    config.diffusion.atom_attention_decoder.no_blocks = 2
    config.diffusion.diffusion_transformer.no_blocks = 6
    
    # Enable v2 flash inference features
    config.globals.advanced_conversion = True
    config.diffusion.diffusion_conditioning.advanced_conditioning = True
    config.diffusion.atom_attention_encoder.advanced_conversion = True
    config.diffusion.atom_attention_decoder.advanced_conversion = True
    config.backbone.input_embedder.advanced_conversion = True
    config.backbone.msa.msa_stack.skip_unused_modules = True
    
    return config
