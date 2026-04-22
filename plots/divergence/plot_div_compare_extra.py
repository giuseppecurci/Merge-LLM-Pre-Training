import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

ONE_STEP_TO_TOKENS = 2_097_152
BASE_PATH = "evaluation/plots/div_comparison"
FORMAT_PIC = ".pdf"

# -------------------------
# X-axis transform
# -------------------------
def transform_x(steps):
    return [int(s * ONE_STEP_TO_TOKENS / 1e9) for s in steps], "Tokens (B)"
    
# -------------------------
# Load modelwise divergence
# -------------------------
def load_modelwise(json_path, space_name, metric_name=None):
    """Load divergence for a space (activations/logits/probabilities)."""
    with open(json_path) as f:
        payload = json.load(f)

    steps = np.asarray(payload["steps"], dtype=float)
    data = payload["data"]

    if space_name == "activations":
        act_data = data["activations"]
        all_weights = []

        for weight_name, steps_samples in act_data.items():
            weight_steps = []
            for step_samples in steps_samples:
                arr = np.asarray(step_samples, dtype=float)
                step_mean = np.nanmean(arr) if arr.size > 0 else np.nan
                weight_steps.append(step_mean)
            all_weights.append(weight_steps)

        stacked = np.stack(all_weights, axis=0)
        valid = np.isfinite(stacked)
        y = np.nansum(stacked, axis=0) / np.maximum(valid.sum(axis=0), 1)
        y[valid.sum(axis=0) == 0] = np.nan

    else:
        # logits / probabilities
        metric_data = data[space_name][metric_name]
        arr = np.asarray(metric_data, dtype=float)
        y = np.nanmean(arr, axis=1)

    return steps, y

# -------------------------
# Plot comparison
# -------------------------
def plot_comparison(curves, xlabel, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))

    # Plot raw curves
    for label, (x_vals, y_vals) in curves.items():
        ax.plot(x_vals, y_vals, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left")

    # Plot percentage change
    ax2 = ax.twinx()
    for _, (x_vals, y_vals) in curves.items():
        finite_idx = np.where(np.isfinite(y_vals))[0]
        if len(finite_idx) > 0:
            idx0 = finite_idx[0]
            y0 = y_vals[idx0]
            if y0 != 0:
                pct = (y_vals - y0) / y0 * 100
                ax2.plot(x_vals, pct, linestyle="--", alpha=0.4)
    ax2.set_ylabel("Δ from start (%)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsons", nargs="+", help="Paths to divergence.json files", required=True)
    parser.add_argument("--labels", nargs="+", required=True, help="Labels for each JSON")
    parser.add_argument("--space", choices=["activations", "logits", "probabilities"], required=True)
    parser.add_argument("--metric", choices=["cosine", "kl"], help="Metric (required for logits/probabilities)")
    parser.add_argument("--out", required=True, help="Output plot path")
    args = parser.parse_args()

    if len(args.jsons) != len(args.labels):
        raise ValueError("Number of JSONs must match number of labels")

    curves = {}
    for json_path, label in zip(args.jsons, args.labels):
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(json_path)

        if args.space == "activations":
            steps, y = load_modelwise(json_path, "activations")
            ylabel = "Activations - Cosine Distance"
        else:
            if args.metric is None:
                raise ValueError("Metric must be set for logits/probabilities")
            steps, y = load_modelwise(json_path, args.space, args.metric)
            metric_label = "Cosine Distance" if args.metric == "cosine" else "KL Divergence"
            ylabel = f"{args.space.capitalize()} - {metric_label}"

        x_vals, _ = transform_x(steps)
        curves[label] = (x_vals, y)

    out_path = os.path.join(BASE_PATH, args.out)
    plot_comparison(curves, xlabel="Tokens (B)", ylabel=ylabel, out_path=out_path)

if __name__ == "__main__":
    main()