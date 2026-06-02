import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.asarray(x, dtype=float)

    if p == 0.0:
        pattern = np.ones_like(x, dtype=float)
        return x.copy(), pattern

    keep_prob = 1.0 - p
    random_vals = rng.random(x.shape) if rng is not None else np.random.random(x.shape)
    mask = random_vals < keep_prob

    scale = 1.0 / keep_prob
    dropout_pattern = mask.astype(float) * scale
    output = x * dropout_pattern

    return output, dropout_pattern