import numpy as np

def generator(z, W, b):
    """
    Returns: np.ndarray of shape (batch_size, output_dim)
    """
    return np.tanh(np.dot(z, W) + b)