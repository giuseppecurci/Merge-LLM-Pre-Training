from pathlib import Path

def compute_linear_weights(checkpoints: dict[int, Path]):
    """
    Linear averaging: all weights = 1
    """
    return {step: 1.0 for step in checkpoints}

def compute_ema_weights(checkpoints: dict[int, Path], alpha: float):
    """
    EMA weights where checkpoints are ordered chronologically.
    Recurrence:
        w_i = alpha
        w_{i-1} = alpha * (1-alpha)
        ...
        w_1 = (1-alpha)^(n-1)
    """
    steps = sorted(checkpoints)
    n = len(steps)

    w = {}
    for idx, step in enumerate(steps):
        if idx == n - 1:  # newest model
            w[step] = alpha
        else:
            power = (n - idx - 1)
            if idx == 0:
                w[step] = (1 - alpha)**power
            else:
                w[step] = alpha * (1 - alpha)**power
    return w

def compute_wma_weights(checkpoints: dict[int, Path]):
    """
    Weighted moving average:
    weights increase linearly with checkpoint order, starting from 1.
    """
    steps = sorted(checkpoints)
    return {step: (i+1) for i, step in enumerate(steps)}