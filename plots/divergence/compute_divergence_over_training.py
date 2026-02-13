import os
import json
import argparse
from collections import defaultdict
import numpy as np
import torch
from safetensors.torch import safe_open
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from tqdm import tqdm
import pprint
import yaml

BASE_DIR = os.path.dirname(__file__)
ONE_STEP_TO_TOKENS = 2_097_152

# -------------------------
# Get Checkpoints
# -------------------------

def resolve_checkpoint_paths(base_path: str, steps: list[int]) -> dict[int, Path]:
    """
    Resolve checkpoint directories for each step.
    Tries exact path first, otherwise searches for similar matches.
    Returns {step: resolved_dir}.
    """
    base = Path(base_path)
    resolved = {}

    for step in steps:
        expected = base / f"step={step}" / "hf/model.safetensors"
        if expected.exists():
            resolved[step] = expected
            continue

        # Search for similar patterns (e.g. step=297999-save)
        pattern = f"step={step}"
        candidates = [p / "hf/model.safetensors" for p in base.iterdir() if p.is_dir() and p.name.startswith(pattern)]

        if len(candidates) == 1:
            warnings.warn(
                f"Exact path '{expected}' not found; using '{candidates[0]}' instead."
            )
            resolved[step] = candidates[0]
        elif len(candidates) > 1:
            raise FileNotFoundError(
                f"Multiple matches for '{expected}': {[str(c) for c in candidates]}. "
            )
        else:
            warnings.warn(
                f"Checkpoint directory for step={step} not found "
                f"and no similar matches in {base}."
            )

    assert len(resolved) >= 2, f"Found only {len(resolved)} checkpoints, but need at least 2 for {base_path}"
    
    return resolved

# -------------------------
# Weight streaming
# -------------------------

def iter_weights(model_path, device):
    with safe_open(model_path, framework="pt", device=device) as f:
        for key in f.keys():
            yield key, f.get_tensor(key)

# -------------------------
# Metrics
# -------------------------

def rms_delta(t1, t2):
    return float(torch.sqrt(torch.mean((t1 - t2) ** 2)))

# -------------------------
# Divergence computation
# -------------------------

def compute_divergence(path_a, path_b, device):
    weights_a = {}
    for k, t in iter_weights(path_a, device):
        weights_a[k] = t

    results = defaultdict(lambda: defaultdict(list))

    for k, t_b in iter_weights(path_b, device):
        if k not in weights_a:
            warnings.warn(f"missing weight {k} from {path_b} w.r.t. {path_a}")
            continue
        t_a = weights_a[k]
        if t_a.shape != t_b.shape:
            warnings.warn(f"shape mismatch weight {k} from {path_b} w.r.t. {path_a}")

        results["rms"][k].append(rms_delta(t_a, t_b))

    return results

# -------------------------
# Sliding window analysis
# -------------------------

def sliding_window_incremental(step_to_path, window_merge, device):

    steps = sorted(step_to_path.keys())

    all_window_starts = [
        s for s in steps
        if any(t >= s + window_merge for t in steps)
    ]


    print(f"\nTotal windows: {len(all_window_starts)}")
    print(f"Windows to compute: {len(all_window_starts)}")

    aggregate = defaultdict(lambda: defaultdict(list))
    final_steps = []
    
    pbar = tqdm(all_window_starts, desc="Computing windows")

    for s1 in pbar:
        target_step = s1 + window_merge

        # find first available step >= target_step
        s2_candidates = [s for s in steps if s >= target_step]

        if not s2_candidates:
            # no valid following checkpoint
            continue

        s2 = s2_candidates[0]

        div = compute_divergence(
            step_to_path[s1],
            step_to_path[s2],
            device,
        )

        for metric, layers in div.items():
            for layer, vals in layers.items():
                aggregate[metric][layer].append(float(np.mean(vals)))

        final_steps.append(s1)

    # sort consistently
    order = np.argsort(final_steps)
    final_steps = [int(final_steps[i]) for i in order]

    for metric in aggregate:
        for layer in aggregate[metric]:
            arr = np.array(aggregate[metric][layer], dtype=float)
            aggregate[metric][layer] = [float(arr[i]) for i in order]

    return aggregate, final_steps

# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-all", action="store_true", help="Recompute and re-plot even if JSON exists")
    parser.add_argument("--device", type=int, default=-1, help="-1:cpu")
    args = parser.parse_args()

    if args.tokens == args.steps:
        raise ValueError("Exactly one of --tokens or --steps must be set")
    
    yaml_path = os.path.join(BASE_DIR, "plot_div.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    ckpts_paths = cfg["ckpts_paths"]
    not_found_ckpts = []
    for ckpt_entry in ckpts_paths:
        path = ckpt_entry["path"]
        if not os.path.exists(path):
            not_found_ckpts.append(path)
    
    if not_found_ckpts:
        print("Not found ckpts:")
        for not_found_ckpt in not_found_ckpts:
            print(f"- {not_found_ckpt}")
        raise FileNotFoundError(f"Checkpoint paths not found")

    for ckpt_entry in ckpts_paths:
        ckpt_path = ckpt_entry["path"]

        model_name = os.path.basename(os.path.dirname(os.path.normpath(ckpt_path)))
        div_experiments = ckpt_entry["div_experiments"]

        for div_exp_dict in div_experiments:
            # Each dict has one key = div_exp_name
            div_exp_name = list(div_exp_dict.keys())[0]
            exp_cfg = div_exp_dict[div_exp_name]

            # Set defaults
            device = exp_cfg.get("device")
            if device is None:
                if args.device == -1:
                    device = "cpu"
                else:
                    device = f"cuda:{device}"
            start_step = exp_cfg["start-step"]
            end_step = exp_cfg["end-step"]
            range_step = exp_cfg["range"]
            window_merge = exp_cfg["window-merge"]

            # Step range
            step_range = [
                step for step in range(start_step, end_step + range_step, range_step)
                if step <= end_step
            ]

            # Out folder for this div experiment
            out_path = os.path.join(BASE_DIR, model_name, div_exp_name)
            os.makedirs(out_path, exist_ok=True)

            print("\n=== Configuration ===")
            pprint.pprint({
                "ckpts_path": ckpt_path,
                "device": device,
                "start_step": start_step,
                "end_step": end_step,
                "range_step": range_step,
                "window_merge": window_merge,
                "num_steps": len(step_range),
                "div_exp": div_exp_name,
                "out_path": out_path
            })

            # Resolve checkpoints
            print("\nResolving checkpoint paths...")
            ckpts = resolve_checkpoint_paths(ckpt_path, step_range)
            print(f"Retrieved {len(ckpts)} checkpoints:")
            for step, path in ckpts.items():
                print(f"  step {step}: {path}")

            json_path = os.path.join(out_path, "divergence.json")
            
            print("\nStarting full sliding-window divergence computation...")
            rms, steps = sliding_window_incremental(
                ckpts,
                window_merge,
                device)

            with open(json_path, "w") as f:
                json.dump({
                    "window": window_merge,
                    "steps": steps,
                    "rms": rms
                    },
                    f,
                    indent=2,
                )

            print(f"\nAll outputs written to: {out_path}")
            print("="*50)

if __name__ == "__main__":
    main()
