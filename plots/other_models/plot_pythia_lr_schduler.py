import numpy as np, math
import matplotlib.pyplot as plt
import os

def lr_schedule(lr0, iters=143000, warmup=0.01, min_ratio=0.1):
    warmup_iters = int(warmup * iters)
    min_lr = lr0 * min_ratio
    lrs = []
    for t in range(iters):
        if t < warmup_iters:
            lr = lr0 * (t / warmup_iters)
        else:
            progress = (t - warmup_iters) / (iters - warmup_iters)
            lr = min_lr + 0.5 * (lr0 - min_lr) * (1 + math.cos(math.pi * progress))
        lrs.append(lr)
    return np.array(lrs)

# ------------------------------------------------------------
# Range computation
# ------------------------------------------------------------

def compute_range(last_step, step_range, N):
    return [last_step - i * step_range for i in range(N)]


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)

TOTAL_ITERS = 143000
BATCH_SIZE = 2_097_152

N = 15
STEP_RANGE = 2000

LAST_DECAY_STEP = 143_000
LAST_STABLE_STEP = 60_000

decay_steps = compute_range(LAST_DECAY_STEP, STEP_RANGE, N)
stable_steps = compute_range(LAST_STABLE_STEP, STEP_RANGE, N)

# Convert to tokens (billions)
steps_tokens = np.arange(TOTAL_ITERS) * BATCH_SIZE / 1e9
decay_tokens = np.array(decay_steps) * BATCH_SIZE / 1e9
stable_tokens = np.array(stable_steps) * BATCH_SIZE / 1e9


# ------------------------------------------------------------
# Schedules
# ------------------------------------------------------------

schedules = {
    "2.8B (lr=1.6e-4)": lr_schedule(0.00016, iters=TOTAL_ITERS),
    "6.9B-12B (lr=1.2e-4)": lr_schedule(0.00012, iters=TOTAL_ITERS),
}


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure(figsize=(8, 4))

for label, lrs in schedules.items():
    plt.plot(steps_tokens, lrs, label=label)

# vertical lines for decay region
plt.axvline(x=decay_tokens[0], linestyle="--", alpha=0.3, color="green")
plt.axvline(x=decay_tokens[-1], linestyle="--", alpha=0.3, color="green")

# vertical lines for stable region
plt.axvline(x=stable_tokens[0], linestyle="--", alpha=0.3, color="purple")
plt.axvline(x=stable_tokens[-1], linestyle="--", alpha=0.3, color="purple")

plt.xlabel("Tokens (B)", fontsize=14)
plt.ylabel("Learning rate", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "pythia_lr_schduler.pdf"))