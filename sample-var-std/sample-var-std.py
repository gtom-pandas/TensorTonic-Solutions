import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    x = np.asarray(x, dtype=float)
    mean = np.mean(x)
    var = np.sum((x - mean) ** 2) / (len(x) - 1)
    std = np.sqrt(var)
    return float(var), float(std)