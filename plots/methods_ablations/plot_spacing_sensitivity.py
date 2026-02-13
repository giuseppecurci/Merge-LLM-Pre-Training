import matplotlib.pyplot as plt
import numpy as np
import os 

BASE_DIR = os.path.dirname(__file__)

# Tokens
tokens = [629, 1342, 2306]
x = np.arange(len(tokens))

# Data (same as before)
data = {
    "Decay": {
        "4B": {
            "Linear": [53.8, 55.3, 57.3],
            "TIES":   [54.1, 55.7, 57.7],
        },
        "8B": {
            "Linear": [53.7, 55.4, 57.4],
            "TIES":   [53.3, 55.3, 57.3],
        },
    },
    "Stable": {
        "4B": {
            "Linear": [53.7, 54.7, 56.4], 
            "TIES":   [53.3, 54.4, 56.3], 
        },
        "8B": {
            "Linear": [53.5, 54.7, 56.5], 
            "TIES":   [53.1, 54.1, 55.4], 
        },
    },
}

fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)

bar_w = 0.18
offsets = {
    ("Linear", "4B"): -1.5 * bar_w,
    ("Linear", "8B"): -0.5 * bar_w,
    ("TIES",   "4B"):  0.5 * bar_w,
    ("TIES",   "8B"):  1.5 * bar_w,
}


colors = {
    "Linear": "#4C72B0",
    "TIES": "#DD8452",
}

for ax, stage in zip(axes, ["Decay", "Stable"]):

    # ---- draw bars ----
    for method in ["Linear", "TIES"]:
        for spacing in ["4B", "8B"]:
            vals = data[stage][spacing][method]
            bars = ax.bar(
                x + offsets[(method, spacing)],
                vals,
                bar_w,
                color=colors[method],
                label=method if spacing == "4B" else None,
            )

            # spacing label INSIDE the bar (keep here)
            for b, h in zip(bars, vals):
                x_center = b.get_x() + b.get_width() / 2
                ax.text(
                    x_center,
                    h - 0.35,
                    spacing,
                    ha="center",
                    va="top",
                    fontsize=7,
                    color="white",
                    fontweight="bold",
                )

    # ---- adjacency-aware accuracy labels (ONCE per token) ----
    for j, tok in enumerate(tokens):
        vals = [
            data[stage]["4B"]["Linear"][j],
            data[stage]["8B"]["Linear"][j],
            data[stage]["4B"]["TIES"][j],
            data[stage]["8B"]["TIES"][j],
        ]

        offs = [
            offsets[("Linear", "4B")],
            offsets[("Linear", "8B")],
            offsets[("TIES", "4B")],
            offsets[("TIES", "8B")],
        ]

        start = 0
        while start < len(vals):
            end = start + 1
            while end < len(vals) and vals[end] == vals[start]:
                end += 1

            xs = [x[j] + offs[k] for k in range(start, end)]
            center_x = np.mean(xs)
            h = vals[start]

            ax.text(
                center_x,
                h + 0.08,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

            start = end

    ax.set_xticks(x)
    ax.set_xticklabels(tokens)
    ax.set_xlabel("Tokens (B)")
    ax.set_title(stage)
    ax.set_ylim(52, 58.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

axes[0].set_ylabel("Accuracy (%)")
axes[0].legend(ncol=2, fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "spacing_linear_ties.pdf"))
plt.close()