import os
import json
import argparse
from collections import defaultdict
import numpy as np
import torch
import warnings
from pathlib import Path
from tqdm import tqdm
import pprint
import yaml

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset

# -------------------------
# GLOBAL CONFIG
# -------------------------

BATCH_SIZE = 8
SEED = 42
DATASET_NAME = "hellaswag"
USE_LABEL = True

USE_ALL_TOKENS = False

METHODS = ["activations", "logits", "probabilities"] 

set_seed(SEED)

def step_k_formatter(x, pos):
    return f"{int(round(x / 1000))}"

# -------------------------
# DATASET
# -------------------------

def load_batch(tokenizer):
    ds = load_dataset(DATASET_NAME, split="validation").shuffle(seed=SEED).select(range(BATCH_SIZE))

    texts = []
    
    if USE_LABEL:
        for sample in ds:
            ctx = sample["ctx"]
            label_idx = int(sample["label"])
            ending = sample["endings"][label_idx]
            texts.append(ctx + " " + ending)
    else:
        texts = [ds[i]["ctx"] for i in range(BATCH_SIZE)]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True
    )

    return inputs

# -------------------------
# MODEL LOADING
# -------------------------

def load_model(model_path, device):
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()
    return model

# -------------------------
# HOOKS
# -------------------------

def register_hooks(model):
    activations = {}

    def hook_fn(name):
        def fn(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]  # take hidden states
            if isinstance(out, torch.Tensor):
                activations[name] = out.detach()
        return fn

    handles = []
    for name, module in model.named_modules():
        if "layers." in name:
            handles.append(module.register_forward_hook(hook_fn(name)))

    return activations, handles

# -------------------------
# FORWARD
# -------------------------

def forward_collect(model, inputs, device):
    inputs = {k: v.to(device) for k, v in inputs.items()}

    activations, handles = register_hooks(model)

    with torch.no_grad():
        out = model(**inputs)

    logits = out.logits  # (B, T, V)

    if not USE_ALL_TOKENS:
        logits = logits[:, -1, :]
    else:
        logits = logits.mean(dim=1)

    for h in handles:
        h.remove()

    return logits, activations

# -------------------------
# METRICS
# -------------------------

def cosine_per_sample(a, b):
    a = a.flatten(1)
    b = b.flatten(1)

    dot = (a * b).sum(dim=1)
    na = torch.norm(a, dim=1)
    nb = torch.norm(b, dim=1)

    denom = na * nb

    # avoid division by zero
    zero_mask = denom == 0
    denom = torch.where(zero_mask, torch.ones_like(denom), denom)

    cos = dot / denom

    # clamp for numerical stability
    cos = torch.clamp(cos, -1.0, 1.0)

    distance = 1.0 - cos

    # clamp negative distances (FP issues)
    distance = torch.clamp(distance, min=0.0)

    # scale to [0,1]
    distance = distance / 2.0

    # zero vectors → no divergence
    distance = torch.where(zero_mask, torch.zeros_like(distance), distance)

    return distance.cpu().tolist()

def kl_per_sample(logits_a, logits_b):
    p = torch.softmax(logits_a, dim=-1)
    q = torch.softmax(logits_b, dim=-1)

    kl = (p * (p.log() - q.log())).sum(dim=-1)
    return kl.cpu().tolist()

# -------------------------
# CHECKPOINT RESOLUTION
# -------------------------

def resolve_checkpoint_paths(base_path: str, steps: list[int]) -> dict[int, Path]:
    base = Path(base_path)
    resolved = {}

    for step in steps:
        expected = base / f"step={step}" / "hf"
        if expected.exists():
            resolved[step] = expected
            continue

        pattern = f"step={step}"
        candidates = [
            p / "hf"
            for p in base.iterdir()
            if p.is_dir() and p.name.startswith(pattern)
        ]

        if len(candidates) == 1:
            warnings.warn(f"Using fallback {candidates[0]}")
            resolved[step] = candidates[0]
        elif len(candidates) > 1:
            raise FileNotFoundError(f"Multiple matches for step {step}")
        else:
            warnings.warn(f"No checkpoint found for step {step}")

    assert len(resolved) >= 2
    return resolved

# -------------------------
# DIVERGENCE (NEW CORE)
# -------------------------

def compute_divergence(path_a, path_b, batch_input):
    device_a = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"
    device_b = "cuda:1" if torch.cuda.device_count() > 1 else device_a

    model_a = load_model(path_a, device_a)
    model_b = load_model(path_b, device_b)

    logits_a, acts_a = forward_collect(model_a, batch_input, device_a)
    logits_b, acts_b = forward_collect(model_b, batch_input, device_b)

    logits_a = logits_a.cpu()
    logits_b = logits_b.cpu()

    results = defaultdict(lambda: defaultdict(list))

    # LOGITS
    if "logits" in METHODS:
        results["logits"]["cosine"] = cosine_per_sample(logits_a, logits_b)

    # PROBABILITIES
    if "probabilities" in METHODS:
        results["probabilities"]["kl"] = kl_per_sample(logits_a, logits_b)

    # ACTIVATIONS
    if "activations" in METHODS:
        for layer in acts_a:
            if layer not in acts_b:
                continue

            a = acts_a[layer].cpu()
            b = acts_b[layer].cpu()

            if not USE_ALL_TOKENS:
                a = a[:, -1, :]
                b = b[:, -1, :]
            else:
                a = a.mean(dim=1)
                b = b.mean(dim=1)

            cos = cosine_per_sample(a, b)

            results["activations"][layer] = cos

    return results

# -------------------------
# SLIDING WINDOW
# -------------------------

def sliding_window_incremental(step_to_path, window_merge, batch_input,
                               existing_steps=None, existing_data=None):

    steps = sorted(step_to_path.keys())

    all_window_starts = [
        s for s in steps if any(t >= s + window_merge for t in steps)
    ]

    if existing_steps is None:
        existing_steps = []

    missing = sorted(set(all_window_starts) - set(existing_steps))

    print(f"Total windows: {len(all_window_starts)}")
    print(f"Missing: {len(missing)}")

    if existing_data is None:
        aggregate = defaultdict(lambda: defaultdict(list))
        final_steps = []
    else:
        aggregate = defaultdict(lambda: defaultdict(list))
        for k1 in existing_data:
            for k2 in existing_data[k1]:
                aggregate[k1][k2] = list(existing_data[k1][k2])
        final_steps = list(existing_steps)

    for s1 in tqdm(missing):
        target = s1 + window_merge
        s2_candidates = [s for s in steps if s >= target]

        if not s2_candidates:
            print(f"No candidate found for window starting at {s1}")
            continue

        s2 = s2_candidates[0]

        div = compute_divergence(
            step_to_path[s1],
            step_to_path[s2],
            batch_input=batch_input
        )

        for space, metrics in div.items():
            for key, vals in metrics.items():
                aggregate[space][key].append(vals)

        final_steps.append(s1)

    order = np.argsort(final_steps)
    final_steps = [int(final_steps[i]) for i in order]

    for space in aggregate:
        for key in aggregate[space]:
            arr = aggregate[space][key]
            aggregate[space][key] = [arr[i] for i in order]

    return aggregate, final_steps

# -------------------------
# MAIN
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    # axis control (kept for compatibility with original script)
    parser.add_argument("--out-dir", default="evaluation/plots/analysis/div_extra")
    args = parser.parse_args()

    yaml_path = os.path.join(args.out_dir, "plot_div_extra.yaml")

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    for ckpt_entry in cfg["ckpts_paths"]:
        ckpt_path = ckpt_entry["path"]
        print("\n=== Model Path ===")
        pprint.pprint(ckpt_path)

        model_name = os.path.basename(os.path.dirname(os.path.normpath(ckpt_path)))
        for div_exp_dict in ckpt_entry["div_experiments"]:
            name = list(div_exp_dict.keys())[0]
            exp = div_exp_dict[name]

            if name == "241999-299999-w4k-r2k":
                print("Skipping 241999-299999-w4k-r2k because already done")
                continue

            start = exp["start-step"]
            end = exp["end-step"]
            step_size = exp["range"]
            window = exp["window-merge"]

            step_range = list(range(start, end + step_size, step_size))

            print("\n=== Config ===")
            pprint.pprint(name)

            ckpts = resolve_checkpoint_paths(ckpt_path, step_range)

            any_ckpt = next(iter(ckpts.values()))
            tokenizer = AutoTokenizer.from_pretrained(any_ckpt)
            tokenizer.pad_token = tokenizer.eos_token
            batch_inputs = load_batch(tokenizer)

            out_path = os.path.join(args.out_dir, model_name, name, f"{DATASET_NAME}_label{USE_LABEL}_allTokens{USE_ALL_TOKENS}", "_".join(METHODS))
            os.makedirs(out_path, exist_ok=True)

            json_path = os.path.join(out_path, "divergence_extra.json")

            if os.path.exists(json_path):
                with open(json_path) as f:
                    payload = json.load(f)
                data, steps = sliding_window_incremental(
                    ckpts,
                    window,
                    existing_steps=payload["steps"],
                    existing_data=payload["data"],
                    batch_input=batch_inputs
                )
            else:
                data, steps = sliding_window_incremental(
                    ckpts,
                    window,
                    batch_input=batch_inputs
                )

            with open(json_path, "w") as f:
                json.dump(
                    {
                        "window": window,
                        "steps": steps,
                        "data": data,
                    },
                    f,
                    indent=2,
                )

            print(f"Saved to {json_path}")

if __name__ == "__main__":
    main()