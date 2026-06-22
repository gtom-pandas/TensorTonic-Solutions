import numpy as np

def average_pooling_2d(X, pool_size):
    """
    Apply 2D average pooling with non-overlapping windows.
    """
    X = np.asarray(X, dtype=float)
    H, W = X.shape

    H_out = H // pool_size
    W_out = W // pool_size

    X_trim = X[:H_out * pool_size, :W_out * pool_size]
    X_reshaped = X_trim.reshape(H_out, pool_size, W_out, pool_size)
    out = X_reshaped.mean(axis=(1, 3))

    return out.tolist()