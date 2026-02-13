import numpy as np
import math
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------------
# OLMo-3 7B schedule
# ------------------------------------------------------------

def olmo3_lr_schedule(
    peak_lr=3e-4,
    final_ratio=0.1,
    warmup_steps=2000,
    total_tokens=5.93e12,
    batch_size=4_194_304,
):
    total_steps = int(total_tokens / batch_size)
    final_lr = peak_lr * final_ratio

    lrs = []

    for t in range(total_steps):
        # Linear warmup
        if t < warmup_steps:
            lr = peak_lr * (t / warmup_steps)

        # Modified cosine decay
        else:
            progress = (t - warmup_steps) / (total_steps - warmup_steps)
            lr = final_lr + 0.5 * (peak_lr - final_lr) * (
                1 + math.cos(math.pi * progress)
            )

        lrs.append(lr)

    return np.array(lrs), total_steps


# ------------------------------------------------------------
# Range computation
# ------------------------------------------------------------

def compute_range(last_step, step_range, N):
    return [last_step - i * step_range for i in range(N)]


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)

BATCH_SIZE = 4_194_304
TOTAL_TOKENS = 5.93e12

STEP_RANGE = 1_000
N = 15

LAST_DECAY_STEP = 1_413_000
LAST_STABLE_STEP = 286_000

# Compute schedule
lrs, TOTAL_STEPS = olmo3_lr_schedule(
    peak_lr=3e-4,
    final_ratio=0.1,
    warmup_steps=2000,
    total_tokens=TOTAL_TOKENS,
    batch_size=BATCH_SIZE,
)

# Convert step index to billions of tokens
steps_tokens = np.arange(TOTAL_STEPS) * BATCH_SIZE / 1e9

# Compute checkpoint ranges
decay_steps = compute_range(LAST_DECAY_STEP, STEP_RANGE, N)
stable_steps = compute_range(LAST_STABLE_STEP, STEP_RANGE, N)

decay_tokens = np.array(decay_steps) * BATCH_SIZE / 1e9
stable_tokens = np.array(stable_steps) * BATCH_SIZE / 1e9


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure(figsize=(8, 4))

plt.plot(steps_tokens, lrs, label="OLMo-3 7B (peak 3e-4)")

plt.axvline(x=decay_tokens[-1], linestyle="--", alpha=0.3, color="green")
plt.axvline(x=decay_tokens[0], linestyle="--", alpha=0.3, color="green")

# pseudo-stable region (dotted)
plt.axvline(x=stable_tokens[-1], linestyle="--", alpha=0.3, color="purple")
plt.axvline(x=stable_tokens[0], linestyle="--", alpha=0.3, color="purple")

plt.xlabel("Tokens (B)", fontsize=14)
plt.ylabel("Learning rate", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "olmo3_lr_scheduler.pdf"))