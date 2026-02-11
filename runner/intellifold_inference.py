import argparse
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  ## to avoid memory fragmentation
import gc
import json
import traceback
import numpy as np
import click
from pathlib import Path
from datetime import timedelta
from tqdm import tqdm
import logging

#### Model
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.utils import set_seed

from intellifold.openfold.config import model_config
from intellifold.openfold.inference_config import get_model_config
from intellifold.openfold.v2_flash_inference_config import get_model_config as get_v2_flash_config
from intellifold.openfold.v2_inference_config import get_model_config as get_v2_model_config
from intellifold.openfold.model.model import IntelliFold
from intellifold.openfold.utils.atom_token_conversion import aggregate_fn_advanced as aggregate_fn
from intellifold.openfold.model.confidences import get_summary_confidence, get_full_confidence

#### Data Processing
from intellifold.data.module.inference import get_inference_dataloader, construct_empty_template_features
from intellifold.data.types import MSA, Manifest, Record
from intellifold.data.write.writer import write_cif
from intellifold.data.inference.params import BoltzProcessedInput
from intellifold.data.inference.utils import download, get_cache_path
from intellifold.data.inference.data_tools import check_inputs, compute_msa, process_inputs, check_outputs, compute_similar_sequence
       
logger = logging.getLogger(__name__)
 

def init_logging():
    LOG_FORMAT = "[%(asctime)s] [%(levelname)-4s] [%(filename)s:%(lineno)s:%(funcName)s] %(message)s"
    logging.basicConfig(
        format=LOG_FORMAT,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )
    
def predict_and_save(
    args,
    model,
    input_features,
    record,
    structure,
    out_dir,
    seed,
    ):
    
    """
    Predict and save the results.
    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments.
    model : torch.nn.Module
        The model to use for prediction.
    input_features : dict
        The input features to use for prediction.
    record : Record
        The record to use for prediction.
    structure : str
        The structure to use for prediction.
    out_dir : Path
        The output directory to save the predictions to.
    seed : int
        The random seed to use for prediction.    
    """
    
    output_dir = out_dir / "predictions"
    struct_dir = output_dir / record.id
    
    finished = check_outputs(record, struct_dir, seed, args.num_diffusion_samples, args.output_format)
    
    if finished and not args.override:
        msg = (
            f"Found existing predictions for [{record.id}] with seed [{seed}], "
            "If you wish to override these existing predictions, please set the --override flag."
        )
        logger.info(msg)
        return struct_dir
    
    elif finished and args.override:
        msg = (
            f"Found existing predictions for [{record.id}] with seed [{seed}], "
            "and will be overridden."
        )
        logger.info(msg)
    
    with torch.no_grad():
        outputs = model(
            input_features, 
            diffusion_batch_size=args.num_diffusion_samples
            )
        
    x_predicted = outputs['x_predicted'].cpu()
    plddt = outputs['plddt'].cpu()
    pred_dense_atom_mask = input_features['pred_dense_atom_mask'].cpu()                
    center_idx = input_features['center_idx']
    
    summary_confidences_list = get_summary_confidence(outputs, input_features)
    full_confidences_list    = get_full_confidence(outputs, input_features, structure)
    
    ## save the result
    # Create the output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    struct_dir.mkdir(exist_ok=True)
    
    for i in range(args.num_diffusion_samples):
        
        aggregated_output, _ = aggregate_fn([x_predicted[i:i+1], plddt[i:i+1]], pred_dense_atom_mask)
        aggregated_x_predicted, aggregated_plddt = aggregated_output
        
        ## center atom plddt
        repr_atom_plddt = aggregated_plddt[0, center_idx.cpu()[0]].unsqueeze(0)
        
        # Create path name
        outname = f"{record.id}_seed-{seed}_sample-{i}"
        if args.output_format == "pdb":
            output_path = struct_dir / f"{outname}.pdb"
        else:
            output_path = struct_dir / f"{outname}.cif"
            
        write_cif(structure, record, aggregated_x_predicted, repr_atom_plddt, output_path, output_format=args.output_format)
        
        # Save the summary confidences
        summary_confidences = summary_confidences_list[i]
        summary_confidences['num_recycles'] = args.recycling_iters
        outname = f"{record.id}_seed-{seed}_sample-{i}_summary_confidences.json"
        output_path = struct_dir / outname
        with output_path.open("w") as f:
            json.dump(summary_confidences, f, indent=1)
            
        # Save the full confidences
        full_confidences = full_confidences_list[i]
        outname = f"{record.id}_seed-{seed}_sample-{i}_confidences.json"
        output_path = struct_dir / outname
        with output_path.open("w") as f:
            json.dump(full_confidences, f, indent=1)
    
    return struct_dir

def main(args):
    # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # # DO SOME INITIAL SETUP
    # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
    init_logging()
    seeds = args.seed
    seeds = list(map(int, seeds.split(",")))
    set_seed(seeds[0])
     
    # set timeout to 1800000ms, 30 minutes
    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=False),
                       InitProcessGroupKwargs(timeout=timedelta(seconds=1800000))]
    accelerator = Accelerator(
        kwargs_handlers=kwargs_handlers, 
        log_with='wandb', 
        mixed_precision=args.precision,
        step_scheduler_with_optimizer=False
        )
    
    ######## Cache Data ########
    # Set cache path
    cache = Path(args.cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    # Create output directories
    data = Path(args.data).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir = out_dir / f"{data.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if accelerator.is_main_process:
        # Download necessary data and model
        download(cache, args.model)

    if accelerator.is_main_process:
        # Validate inputs
        data = check_inputs(data)
        if not data:
            logger.warning("No predictions to run, exiting.")
            return
        msg = f"Running predictions for {len(data)} structure"
        msg += "s" if len(data) > 1 else ""
        logger.info(msg)

    # Process inputs
    ccd_path = cache / "ccd.pkl"

    if accelerator.is_main_process:
        process_inputs(
            args,
            data=data,
            out_dir=out_dir,
            ccd_path=ccd_path,
            use_msa_server=args.use_msa_server,
            msa_server_url=args.msa_server_url,
            msa_pairing_strategy=args.msa_pairing_strategy,
            max_msa_seqs=16384,
            use_pairing=not args.no_pairing,
        )
        if args.return_similar_seq:
            compute_similar_sequence(
                out_dir=out_dir,
                processed_polymer_fasta_dir=out_dir / "processed" / "fastas",
                cache_dir=cache)
    ### wait
    accelerator.wait_for_everyone()

    # Load processed data
    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=Manifest.load(processed_dir / "manifest.json"),
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(processed_dir / "constraints")
        if (processed_dir / "constraints").exists()
        else None,
    )
    error_dir = out_dir / "errors"
    error_dir.mkdir(parents=True, exist_ok=True)

    if args.only_run_data_process:
        if accelerator.is_main_process:
            logger.info("Data processing complete. Exiting.")
        return

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CONFIGURE THE DATA
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    pred_loader = get_inference_dataloader(
        args=args,
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        constraints_dir=processed.constraints_dir,
    )
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CONFIGURE THE MODEL
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    if args.model == "v2-flash":
        config = get_v2_flash_config(args)
        checkpoint_path = cache / "intellifold_v2_flash.pt"
    elif args.model == "v2":
        config = get_v2_model_config(args)
        checkpoint_path = cache / "intellifold_v2.pt"
    elif args.model == "v1":
        config = get_model_config(args)
        checkpoint_path = cache / "intellifold_v0.1.0.pt"
    else:
        raise ValueError(f"Invalid model: {args.model}")   

    if accelerator.is_main_process:
        logger.info(f"IntelliFold Model Version: {args.model}")
        logger.info(f'Number Of Diffusion Samples: {args.num_diffusion_samples}')
        logger.info(f"Number Of Sampling Steps: {args.sampling_steps}")
        logger.info(f"Number Of Recycling: {config.backbone.recycling_iters}")
        logger.info(f"Number Of Workers: {args.num_workers}")
        logger.info(f"Number Of Seeds: {len(seeds)}, Seeds: {seeds}")
    
    generator = torch.Generator(device=accelerator.device)
    generator.manual_seed(seeds[0])
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # INITIALIZE THE MODEL
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model = IntelliFold(config,generator = generator)
    if not checkpoint_path.exists():
        logger.info(f"Checkpoint file {checkpoint_path} not found.")
        return
    model_state_dict = torch.load(checkpoint_path, map_location=accelerator.device)
    model.load_state_dict(model_state_dict)
    if accelerator.is_main_process:
        logger.info(f"Successfully loaded model weights from {checkpoint_path}")
    model = accelerator.prepare(model)
    accelerator.wait_for_everyone()
    model.eval()     

    # Create DataLoader
    loader = accelerator.prepare(pred_loader)
    for batch_idx, input_features in tqdm(enumerate(loader), total=len(loader), disable= not accelerator.is_local_main_process, desc=f"Predicting"):      
            
        torch.cuda.empty_cache()
        
        try:
            record = input_features.pop("record")[0]
            structure = input_features.pop("structure")
            input_features['msa'] = F.one_hot(input_features['msa'].long(),num_classes=32).float()
            
            ### reference features woulb be changed in the forward pass, keep the original ones
            ref_keys = [key for key in input_features.keys() if 'ref_' in key]
            original_ref_features = [input_features[key] for key in ref_keys]
                        
            input_features.update(construct_empty_template_features(input_features, device=accelerator.device))
            
        except Exception as e:
            error_msg = f"Error in prediction: {e}{traceback.format_exc()}\n"
            target_msg = f"N_chains: {input_features['N_chains'].item()}, N_tokens: {input_features['N_tokens'].item()}, N_Atoms: {input_features['N_atoms'].item()}, N_alignments: {input_features['N_alignments'].item()}"
            logger.warning(error_msg)
            logger.warning(f"[RANK-{accelerator.process_index}] [{batch_idx+1}/{len(loader)}] Skipping the target [{record.id}]: "
                            f"{target_msg}")
            # Save error info
            with error_dir.joinpath(f"{record.id}.txt").open("w") as f:
                f.write(error_msg)
                f.write(target_msg)
            del input_features
            gc.collect()
            torch.cuda.empty_cache()
            continue
          
        seed_completion = 0    
        for seed in seeds:    
            input_features.update(dict(zip(ref_keys, original_ref_features)))
            
            set_seed(seed)  
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # DO THE FORWARD PASS
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            torch.cuda.empty_cache()
            
            ### set seed
            if hasattr(model, 'generator'):
                model.generator.manual_seed(seed)
            else:
                model.module.generator.manual_seed(seed)   
            
            try:   
                # Run the model
                struct_dir = predict_and_save(
                    args,
                    model=model,
                    input_features=input_features,
                    record=record,
                    structure=structure,
                    out_dir=out_dir,
                    seed=seed,
                    )
                seed_completion += 1
                torch.cuda.empty_cache()
            except Exception as e:
                error_msg = f"Error in prediction: {e}{traceback.format_exc()}\n"
                target_msg = f"N_chains: {input_features['N_chains'].item()}, N_tokens: {input_features['N_tokens'].item()}, N_Atoms: {input_features['N_atoms'].item()}, N_alignments: {input_features['N_alignments'].item()}"
                logger.warning(error_msg)
                logger.warning(f"[RANK-{accelerator.process_index}] [{batch_idx+1}/{len(loader)}] Skipping the target [{record.id}]: "
                               f"{target_msg}")
                # Save error info
                with error_dir.joinpath(f"{record.id}.txt").open("w") as f:
                    f.write(error_msg)
                    f.write(target_msg)
                del original_ref_features
                del input_features
                gc.collect()
                torch.cuda.empty_cache()
                break
            
        if seed_completion == len(seeds):
            logger.info(
                f"[RANK-{accelerator.process_index}] [{batch_idx+1}/{len(loader)}] {record.id} successfully predicted.\n"
                f"Predictions Results saved to {struct_dir}."
                )

    accelerator.wait_for_everyone()
    
    
    if accelerator.is_main_process:
        total_failed = len(os.listdir(error_dir))
        if total_failed > 0:
            logger.warning(f"There are [{total_failed}] targets that failed to predict during inference.")
            logger.warning(f"Failed targets are saved in [{error_dir}].")
            logger.warning(f"Please check the error files in [{error_dir}] for more information.")
        else:
            logger.info("Inference completed successfully.")
            

@click.group()
def intellifold_cli():
    """IntelliFold."""
    return

@click.command()
# DATA DIRECTORIES
@click.argument(
    "data",
    type=str,
)
@click.option(
    "--out_dir",
    type=str,
    help="The path where to save the predictions.",
    default="./",
)
@click.option(
    "--cache",
    type=str,
    help="The directory where to download the data and model. Default is ~/.intellifold, or $INTELLIFOLD_CACHE if set.",
    default=get_cache_path(),
)

# DATA SETTINGS
@click.option(
    '--num_workers', 
    type=int, 
    default=4, 
    help="Number of workers for data loading"
)

# MODEL SETTINGS
@click.option(
    '--precision', 
    type=str, 
    default='bf16', 
    help='Sets precision, lower precision improves runtime performance.'
)

# INFERENCE SETTINGS
@click.option(
    '--seed', 
    type=str,
    default='42', 
    help="Random seed (single int or multiple ints separated by comma, e.g., '42' or '42,43')"
)
@click.option(
    '--recycling_iters', 
    type=int, 
    default=10, 
    help="Number of recycling iterations"
)
@click.option(
    '--num_diffusion_samples', 
    type=int, 
    default=5, 
    help="Batch size for diffusion"
)
@click.option(
    '--sampling_steps', 
    type=int, 
    default=200, 
    help="The number of diffusion sampling steps to use. Default is 200."
)
@click.option(
    "--output_format",
    type=click.Choice(["pdb", "mmcif"]),
    help="The output format to use for the predictions. Default is mmcif.",
    default="mmcif",
)
@click.option(
    "--override",
    is_flag=True,
    help="Whether to override existing found predictions. Default is False.",
)
@click.option(
    "--use_msa_server",
    is_flag=True,
    help="Whether to use the MMSeqs2 server for MSA generation. Default is False.",
)
@click.option(
    "--msa_server_url",
    type=str,
    help="MSA server url. Used only if --use_msa_server is set.",
    default="https://api.colabfold.com",
)
@click.option(
    "--msa_pairing_strategy",
    type=str,
    help="Pairing strategy to use. Used only if --use_msa_server is set. Options are 'greedy' and 'complete'.",
    default="greedy",
)
@click.option(
    "--no_pairing",
    is_flag=True,
    help="Whether to use pairing for Protein Multimer MSA generation. Default is False.",
)
@click.option(
    "--only_run_data_process",
    is_flag=True,
    help="Whether to only run data processing, and not run the model. Default is False.",
)
@click.option(
    "--return_similar_seq",
    is_flag=True,
    help="Whether to return sequences similar to those in the training PDB dataset during inference. Default is False.",
)
# @click.option(
#     "--no_potentials",
#     is_flag=True,
#     help="Whether to not use potentials for steering. Default is False.",
# )
@click.option(
    "--model",
    type=click.Choice(["v1", "v2", "v2-flash"]),
    help="The model to use for prediction. Default is 'v2-flash'.",
    default="v2-flash",
)
def predict(
    data: str,
    out_dir: str,
    cache: str,
    num_workers: int,
    precision: str,
    seed: list,
    recycling_iters: int,
    num_diffusion_samples: int,
    sampling_steps: int,
    output_format: str,
    override: bool,
    use_msa_server: bool,
    msa_server_url: str,
    msa_pairing_strategy: str,
    no_pairing: bool,
    only_run_data_process: bool,
    return_similar_seq: bool,
    model: str,
    # no_potentials: bool,
):
    ## create a argparse.Namespace object
    args = argparse.Namespace(
        data=data,
        out_dir=out_dir,
        cache=cache,
        num_diffusion_samples=num_diffusion_samples,
        sampling_steps=sampling_steps,
        num_workers=num_workers,
        precision=precision,
        recycling_iters=recycling_iters,
        seed=seed,
        output_format=output_format,
        override=override,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        no_pairing=no_pairing,
        only_run_data_process=only_run_data_process,
        return_similar_seq=return_similar_seq,
        model=model,
    )
    main(args=args)

intellifold_cli.add_command(predict)


if __name__ == "__main__":
    predict()
