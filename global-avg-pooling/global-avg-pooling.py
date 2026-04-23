import numpy as np

def global_avg_pool(x):
    """
    23/04/26 Problem of the day :)
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    x = np.asarray(x)

    if x.ndim == 3:
        return x.mean(axis=(1, 2))
    elif x.ndim == 4:
        return x.mean(axis=(2, 3))
    else:
        raise ValueError("Input must have shape (C,H,W) or (N,C,H,W).")