import numpy as np

def discriminator(x, W):
    """
    Returns: np.ndarray of shape (batch, 1) with probabilities
    """
    logits = np.dot(x, W)
    return 1 / (1 + np.exp(-logits))