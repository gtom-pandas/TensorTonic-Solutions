import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).

    (N, D): normalize each feature over axis=0
    (N, C, H, W): normalize each channel over axes (0,2,3)
    """
    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)

    if x.ndim == 2:
        # x: (N, D), gamma/beta: (D,)
        mean = x.mean(axis=0, keepdims=True)                 # (1, D)
        var = ((x - mean) ** 2).mean(axis=0, keepdims=True)  # (1, D)
        x_hat = (x - mean) / np.sqrt(var + eps)              # (N, D)
        return x_hat * gamma + beta

    if x.ndim == 4:
        # x: (N, C, H, W), gamma/beta: (C,)
        mean = x.mean(axis=(0, 2, 3), keepdims=True)                 # (1, C, 1, 1)
        var = ((x - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)  # (1, C, 1, 1)
        x_hat = (x - mean) / np.sqrt(var + eps)                      # (N, C, H, W)

        gamma_b = gamma.reshape(1, -1, 1, 1)
        beta_b = beta.reshape(1, -1, 1, 1)
        return x_hat * gamma_b + beta_b

    raise ValueError("x must have shape (N, D) or (N, C, H, W) -_-")