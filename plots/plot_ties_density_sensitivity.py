import matplotlib.pyplot as plt
import numpy as np
import os 

BASE_DIR = os.path.dirname(__file__)

# Tokens
tokens = [629, 1342, 2306]
x = np.arange(len(tokens))

# TIES densities and methods
densities = ["0.2", "0.4", "0.6", "0.8", "1.0"]

# Made-up data for missing values (slightly adjusted)
data = {
    "Stable": {
        "Linear": [53.7, 54.7, 56.4],
        "TIES": {
            "0.2": [51.6, 52.1, 54.0], 
            "0.4": [52.1, 53.2, 55.1], 
            "0.6": [53.0, 54.0, 55.7], 
            "0.8": [53.3, 54.4, 56.3], 
            "1.0": [53.3, 54.6, 56.2], 
        },
    },
    "Decay": {
        "Linear": [53.8, 55.3, 57.3],
        "TIES": {
            "0.2": [53.2, 55.5, 57.2],
            "0.4": [53.5, 55.8, 57.4],
            "0.6": [53.7, 55.5, 57.4],
            "0.8": [54.1, 55.7, 57.7],
            "1.0": [54.1, 55.2, 57.7],
        },
    }
}

fig, axes = plt.subplots(1, 2, figsize=(13, 3), sharey=True)

bar_w = 0.12
n_bars = 6  # Linear + 5 TIES densities
offsets = np.linspace(-2.5*bar_w, 2.5*bar_w, n_bars)
colors = ["#4C72B0"] + ["#DD8452"]*5  # Linear blue, TIES orange

for ax, stage in zip(axes, ["Decay", "Stable"]):
    # Collect all heights for each group (token)
    all_vals = []
    for i, method in enumerate(["Linear"] + densities):
        if method == "Linear":
            all_vals.append(data[stage]["Linear"])
        else:
            all_vals.append(data[stage]["TIES"][method])
    all_vals = np.array(all_vals)  # shape: (6 bars, 3 tokens)

    for i, method in enumerate(["Linear"] + densities):
        vals = all_vals[i]
        bars = ax.bar(
            x + offsets[i],
            vals,
            bar_w,
            color=colors[i],
            label="Linear" if method=="Linear" else "TIES" if i==1 else None
        )

        # Inside bar: density labels (skip Linear)
        if method != "Linear":
            for idx, b in enumerate(bars):
                h = b.get_height()
                ax.text(
                    b.get_x() + b.get_width()/2,
                    h - 0.25,
                    f"{method}",
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="white",
                    fontweight="bold"
                )

    # On top of bars: only one number per unique height
    for j in range(len(tokens)):
        vals_j = all_vals[:, j]   # shape: (n_bars,)
        
        start = 0
        while start < len(vals_j):
            end = start + 1
            # extend while adjacent bars have same value
            while end < len(vals_j) and vals_j[end] == vals_j[start]:
                end += 1

            # bars [start:end] form one group
            xs = x[j] + offsets[start:end]
            center_x = xs.mean()

            ax.text(
                center_x,
                vals_j[start] + 0.1,
                f"{vals_j[start]:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

            start = end


    ax.set_xticks(x)
    ax.set_xticklabels(tokens, fontsize=14)
    ax.set_xlabel("Tokens (B)", fontsize=14)
    ax.set_title(stage, fontsize=14)
    ax.set_ylim(50.5, 58.5)
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

axes[0].set_ylabel("Accuracy (%)", fontsize=14)
axes[0].legend(ncol=2, fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "ties_density_barplot.pdf"))
plt.close()
