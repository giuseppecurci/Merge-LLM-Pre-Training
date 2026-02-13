from pathlib import Path
import os
import sys
sys.path.append(str(Path(__file__).parent))

from get_checkpoints import resolve_checkpoint_paths
from utils import *
import argparse
import random

MERGED_MODEL_FOLDER = "merged_model"

def is_random(inject_noise, ckpt_dropout):
    if inject_noise or ckpt_dropout > 0.0:
        return True
    return False

def needs_ensemble(seeds, inject_noise, ckpt_dropout, ensemble_merge_strategy):
    if ensemble_merge_strategy:
        if len(seeds) <= 1:
            raise ValueError(f"Provided ensemble merge strategy {ensemble_merge_strategy} with not enough seeds: {len(seeds)}. Need >1")
        if is_random(inject_noise, ckpt_dropout):
            return True
        else:
            raise ValueError("Can't ensemble merges with no random strategies")
    return False

def build_ensemble_name(strategy, inject_noise, noise_scale, ckpt_dropout, n):
    parts = [strategy]

    if inject_noise:
        parts.append(f"noisy_scale{noise_scale}")
    if ckpt_dropout > 0.0:
        parts.append(f"ckptDropout{ckpt_dropout}")

    parts.append(f"{n}models")
    return "_".join(parts)

def prepare_ensemble_merge(
        merge_dirs: list[str],
        checkpoints_path: str,
        base_out_dir: str,
        start_step: int,
        end_step: int,
        range_step: int,
        ensemble_merge_strategy: str,
        seed: int,
        inject_noise: bool,
        noise_scale: float,
        ckpt_dropout: float
    ):

    model_folder_name = os.path.split(os.path.dirname(checkpoints_path))[-1]

    ensemble_name = build_ensemble_name(
        ensemble_merge_strategy,
        inject_noise,
        noise_scale,
        ckpt_dropout,
        len(merge_dirs),
    )

    out_dir = os.path.join(
        base_out_dir,
        model_folder_name,
        ensemble_name,
        f"{start_step}-{end_step}"
    )

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # convert to dict just for compatibility
    ensemble_checkpoints = {
        i:os.path.join(d, "model.safetensors")
        for i,d in enumerate(merge_dirs)
    }

    yaml_path, already_exists = write_mergekit_yaml(
        ensemble_checkpoints,
        out_dir,
        merge_strategy=ensemble_merge_strategy,
        dtype="float16"
    )

    if already_exists:
        print(f"[WARNING] Mergekit YAML for {ensemble_name}{ensemble_merge_strategy} already exists at: {yaml_path.resolve()}")

    command = [
        "mergekit-pytorch",
        "$YAML_PATH",
        "$OUT_DIR",
        f"--random-seed {seed}\n",
    ]

    command = " \\\n    ".join(command) + "\n"

    command_path = os.path.join(out_dir, "merge.sh")
    with open(command_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("set -e\n")
        f.write("set -o pipefail\n\n")

        f.write(
            f"echo 'Running {ensemble_name} on {model_folder_name} "
            f"with {len(ensemble_checkpoints)} checkpoints "
            f"from training steps {start_step}-{end_step} (range:{range_step})'\n\n"
        )

        f.write(f"YAML_PATH='{yaml_path}'\n")
        f.write(f"OUT_DIR='{out_dir}'\n\n")

        f.write(command)

        f.write(f"mkdir $OUT_DIR/{MERGED_MODEL_FOLDER} \n")
        f.write(f"mv \\\n  $OUT_DIR/model.safetensors \\\n  $OUT_DIR/{MERGED_MODEL_FOLDER}\n\n")

        any_checkpoint_hf_folder = Path(ensemble_checkpoints[0]).parent

        f.write(f"CP_SRC='{any_checkpoint_hf_folder}'\n")
        f.write("cp \\\n")
        for fname in [
            "config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "tokenizer.model",
        ]:
            f.write(f"  $CP_SRC/{fname} \\\n")
        f.write(f"  $OUT_DIR/{MERGED_MODEL_FOLDER}\n")

    os.chmod(command_path, 0o755)

def prepare_single_merge(
        checkpoints_path: str,
        out_dir: str,
        start_step: int,
        end_step: int,
        merge_strategy: str,
        seed: int,
        async_write: bool,
        num_threads: int,
        write_threads: int,
        range_step: int = 2_000,
        dtype: str = "float16",
        alpha: float | None = None,
        pull_secondary_model: float | None = None,
        device: int | None = None,
        inject_noise: bool | None = None,
        noise_scale: int = 1e-3,
        ckpt_dropout: float = 0.0,
        lambda_weight: float = None,
        density: float = None
    ):  
    
    random.seed(seed)
    step_range = [step for step in range(start_step, end_step + range_step, range_step) if step <= end_step]
    if ckpt_dropout > 0.0:
        total = len(step_range)
        keep_n = int(total - total * ckpt_dropout)
        step_range = random.sample(step_range, keep_n)
    checkpoints = resolve_checkpoint_paths(checkpoints_path, step_range)
    num_checkpoints = len(checkpoints)
    print(f"[INFO] Retrieved {num_checkpoints} checkpoint{'s' if num_checkpoints != 1 else ''} "
      f"from steps {start_step} to {end_step}")
    validate_inputs(start_step, end_step, range_step, 
                    merge_strategy, alpha, pull_secondary_model, inject_noise,
                    lambda_weight, density)
    
    model_folder_name = os.path.split(os.path.dirname(checkpoints_path))[-1] \
        if ("pythia" not in checkpoints_path.lower() and \
            "smollm" not in checkpoints_path.lower() and \
            "olmo" not in checkpoints_path.lower()) else os.path.split(checkpoints_path)[-1]
    out_dir = os.path.join(out_dir, model_folder_name)
    out_dir += f"/{merge_strategy}"
    if alpha:
        out_dir += f"_alpha{alpha}"
    if pull_secondary_model:
        out_dir += f"_pull_model{pull_secondary_model}"
    if is_random(inject_noise, ckpt_dropout):
        out_dir += f"_{seed}"
    if inject_noise:
        out_dir += f"_noisy_scale{noise_scale}"
    if ckpt_dropout > 0.0:
        out_dir += f"_ckptDropout{ckpt_dropout}"
    if merge_strategy.startswith("ties"):
        out_dir += f"_lambda{lambda_weight}_density{density}"
        
    out_dir += f"/{start_step}-{end_step}" 
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Merge folder created at: {out.resolve()}")

    yaml_path, already_exists = write_mergekit_yaml(
        checkpoints,
        out_dir,
        merge_strategy=merge_strategy,
        pull_secondary_model=pull_secondary_model,
        alpha=alpha,
        dtype=dtype,
        inject_noise=inject_noise,
        noise_scale=noise_scale,
        lambda_weight = lambda_weight,
        density = density
    )
    if already_exists:
        print(f"[WARNING] Mergekit YAML already exists at: {yaml_path.resolve()}")
        return os.path.join(out_dir, MERGED_MODEL_FOLDER) 
    else:
        print(f"[INFO] Mergekit YAML written to: {yaml_path.resolve()}")
    
    merge_multi_stage = merge_strategy == "slerp"

    command = [
        "mergekit-pytorch",
        "$YAML_PATH",
        "--out-path $OUT_DIR" if merge_multi_stage else "$OUT_DIR",
        f"--write-threads {write_threads}",
        f"--num-threads {num_threads}"
    ]

    if async_write:
        command.append("--async-write")

    if device is not None:
        command.extend(["--cuda", f"--device cuda:{device}"])
    if merge_multi_stage:
        command[0] = "mergekit-multi"
        command.append("--intermediate-dir $INTERMEDIATE_DIR")
    
    command.append(f"--random-seed {seed}\n")

    command = " \\\n    ".join(command) + "\n"
    command_path = out_dir + "/merge.sh"
    with open(command_path, "w") as f:
        f.write(f"#!/bin/bash\n")
        f.write("set -e\n")
        f.write("set -o pipefail\n\n")

        merge_strat_to_print = os.path.basename(os.path.dirname(out_dir))
        f.write(f"echo 'Running {merge_strat_to_print} on {model_folder_name} with {num_checkpoints} checkpoints "
                f"from training steps {start_step}-{end_step} (range:{range_step})'\n\n")
        
        f.write(f"YAML_PATH='{yaml_path}'\n")
        if not merge_multi_stage:
            f.write(f"OUT_DIR='{out_dir}'\n\n")
        else:
            f.write(f"OUT_DIR='{out_dir}/{MERGED_MODEL_FOLDER}'\n\n")

        if merge_multi_stage:
            intermediate_dir = str(out / "intermediate")
            f.write(f"INTERMEDIATE_DIR='{intermediate_dir}'\n\n")

        f.write(command)

        if not merge_multi_stage:
            f.write(f"mkdir $OUT_DIR/{MERGED_MODEL_FOLDER}\n\n")

            any_checkpoint = os.path.dirname(list(checkpoints.values())[0])

            f.write("mv \\\n")
            f.write("  $OUT_DIR/*safetensors* \\\n")
            f.write(f"  $OUT_DIR/{MERGED_MODEL_FOLDER}\n\n")

            extra_files = [
                "config.json",
                "special_tokens_map.json",
                "tokenizer_config.json",
                "tokenizer.json"
            ]

            if (
                "pythia" not in any_checkpoint.lower() and \
                "smollm" not in any_checkpoint.lower() and \
                "olmo" not in any_checkpoint.lower()
                ):
                extra_files.append("tokenizer.model")

            f.write(f"CP_SRC='{any_checkpoint}'\n")
            f.write("cp \\\n")

            for extra_file in extra_files:
                f.write(f"  $CP_SRC/{extra_file} \\\n")

            f.write(f"  $OUT_DIR/{MERGED_MODEL_FOLDER}\n")

        if merge_multi_stage:
            f.write(f"mv $INTERMEDIATE_DIR/{MERGED_MODEL_FOLDER}/ $OUT_DIR \n\n")  
            f.write(f"rm -rf $INTERMEDIATE_DIR")   

    os.chmod(command_path, 0o755)
    print(f"[INFO] Merge command saved and made exec at: {command_path}")
    return os.path.join(out_dir, MERGED_MODEL_FOLDER) 

def prepare_merge(
        checkpoints_path: str,
        out_dir: str,
        start_step: int,
        end_step: int,
        merge_strategy: str,
        seeds: list[int],
        async_write: bool,
        num_threads: int,
        write_threads: int,
        range_step: int = 2_000,
        dtype: str = "float16",
        alpha: float | None = None,
        pull_secondary_model: float | list[float] | None = None,
        device: int | None = None,
        inject_noise: bool | None = None,
        noise_scale: int = 1e-3,
        ckpt_dropout: float = 0.0,
        lambda_weight: float = None,
        density: float = None,
        ensemble_merge_strategy = None
    ):
    
    do_ensemble = needs_ensemble(
        seeds, inject_noise, ckpt_dropout, ensemble_merge_strategy
    )

    if not do_ensemble:
        seeds = [seeds[0]]

    merge_dirs = []

    for s in seeds:
        out_dir_s = prepare_single_merge(
            async_write = async_write,
            num_threads =  num_threads,
            write_threads = write_threads,
            checkpoints_path=checkpoints_path,
            out_dir=out_dir,
            start_step=start_step,
            end_step=end_step,
            merge_strategy=merge_strategy,
            seed=s,
            range_step=range_step,
            dtype=dtype,
            alpha=alpha,
            pull_secondary_model=pull_secondary_model,
            device=device,
            inject_noise=inject_noise,
            noise_scale=noise_scale,
            lambda_weight=lambda_weight,
            density=density,
            ckpt_dropout=ckpt_dropout
        )
        merge_dirs.append(out_dir_s)
        print("="*50)

    if do_ensemble:
        prepare_ensemble_merge(
            merge_dirs=merge_dirs,
            checkpoints_path=checkpoints_path,
            base_out_dir=out_dir,
            start_step=start_step,
            end_step=end_step,
            range_step=range_step,
            ensemble_merge_strategy=ensemble_merge_strategy,
            seed=0, 
            inject_noise=inject_noise,
            noise_scale=noise_scale,
            ckpt_dropout=ckpt_dropout,
        )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a mergekit YAML for model merging."
    )
    parser.add_argument(
        "--ckpts-path", type=str, required=True,
        help="Base path containing checkpoints."
    )
    parser.add_argument(
        "--out-dir", type=str, default="model_merging/merged_models",
        help="Output folder where mergekit YAML will be created."
    )
    parser.add_argument(
        "--async-write", action="store_true", 
        help="Write output shards asynchronously"
    )
    parser.add_argument(
        "--write-threads", type=int, default=4,
        help="Number of threads to use for asynchronous writes  [default: 1]"
    )
    parser.add_argument(
        "--num-threads", type=int, default=4,
        help="Number of threads to use for parallel CPU operations"
    )
    parser.add_argument(
        "--start-step", type=int, required=True,
        help="First checkpoint step to include."
    )
    parser.add_argument(
        "--end-step", type=int, required=True,
        help="Last checkpoint step to include."
    )
    parser.add_argument(
        "--range-step", type=int, default=2000,
        help="Step increment between checkpoints (default: 2000)."
    )
    parser.add_argument(
        "--dtype", type=str, default="float16",
        help="Dtype to set in the YAML file (default: float16)."
    )
    parser.add_argument(
        "--merge-strategy", type=str,
        default=None,
        help="Merging strategy one of ema, wma or any available in mergekit"
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Alpha value for EMA; required if \"ema\" in merge_mode."
    )
    parser.add_argument(
        "--pull-secondary-model",
        type=float,
        help="Float or comma-separated list of floats controlling pull strength toward the new model",
        )
    parser.add_argument(
        "--device", type=int,
        help="CUDA device. If omitted, CPU mode is used."
    )
    parser.add_argument(
        "--inject-noise", action="store_true", default=False,
        help="Whether to add model-layer-wise noise to weights during merging. Now works only with linear."
    )
    parser.add_argument(
        "--noise-scale", type=float, default=1e-3,
        help="Global small float to regulate noise injection"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="*", default=[0, 39, 201],
        help="Multiple seeds for stochastic merges"
    )
    parser.add_argument(
        "--ckpt-dropout", type=float, default=0.0,
        help="Probability to drop a checkpoint"
    )
    parser.add_argument(
        "--density", type=float, default=0.2,
        help="Fraction of weights to retain in each sparsified task vector per model"
    )
    parser.add_argument(
        "--lambda-weight", type=float, default=1,
        help="Multiplier of merged task vector"
    )
    parser.add_argument(
        "--ensemble-merge-strategy", type=str, default=None,
        help="Merge strategy for final ensemble merge"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    prepare_merge(
        checkpoints_path=args.ckpts_path,
        out_dir=args.out_dir,
        start_step=args.start_step,
        end_step=args.end_step,
        range_step=args.range_step,
        dtype=args.dtype,
        merge_strategy=args.merge_strategy,
        alpha=args.alpha,
        pull_secondary_model=args.pull_secondary_model,
        device=args.device,
        inject_noise=args.inject_noise,
        noise_scale=args.noise_scale,
        seeds=args.seeds,
        ckpt_dropout=args.ckpt_dropout,
        lambda_weight=args.lambda_weight,
        density=args.density,
        ensemble_merge_strategy=args.ensemble_merge_strategy,
        async_write = args.async_write,
        num_threads = args.num_threads,
        write_threads = args.write_threads
    )