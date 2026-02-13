import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os 

ONE_STEP_TO_TOKENS = 2_097_152
BASE_PATH = os.path.dirname(__file__)
FORMAT_PIC = ".pdf"

# -------------------------
# X-axis transform
# -------------------------

def transform_x(steps):
    return [int(s * ONE_STEP_TO_TOKENS / 1e9) for s in steps], "Tokens (B)"

# -------------------------
# Load modelwise divergence
# -------------------------

def load_divergence(json_path):
    with open(json_path) as f:
        payload = json.load(f)

    steps = np.asarray(payload["steps"], dtype=float)
    layers_rms = payload["rms"]

    stacked = np.stack(
        [np.asarray(v, dtype=float) for v in layers_rms.values()],
        axis=0
    )

    valid = np.isfinite(stacked)
    summed = np.nansum(stacked, axis=0)
    counts = valid.sum(axis=0)

    y = np.zeros_like(summed)
    mask = counts > 0
    y[mask] = summed[mask] / counts[mask]
    y[~mask] = np.nan

    return steps, y


# -------------------------
# Plot comparison
# -------------------------

def plot_comparison(curves, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))

    for label, (steps, y) in curves.items():
        x_vals, xlabel = transform_x(steps)
        ax.plot(x_vals, y, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("RMS")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left")

    # percentage change (relative to each curve's start)
    ax2 = ax.twinx()
    for _, (steps, y) in curves.items():
        idx0 = np.where(np.isfinite(y))[0][0]
        y0 = y[idx0]
        pct = (y - y0) / y0 * 100
        x_vals, _ = transform_x(steps)
        ax2.plot(x_vals, pct, linestyle="--", alpha=0.4)

    ax2.set_ylabel("Δ from start (%)")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsons", nargs="+", help="Paths to divergence.json files")
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Labels corresponding to each JSON"
    )
    parser.add_argument("--out", required=True, help="Output plot path")

    args = parser.parse_args()

    if len(args.jsons) != len(args.labels):
        raise ValueError("Number of JSONs must match number of labels")

    out_folder = os.path.join(BASE_PATH, args.out)
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, f"modelwise_div_rms{FORMAT_PIC}")

    curves = {}

    for json_path, label in zip(args.jsons, args.labels):
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(json_path)

        steps, y = load_divergence(json_path)
        curves[label] = (steps, y)

    plot_comparison(curves, out_path)


if __name__ == "__main__":
    main()