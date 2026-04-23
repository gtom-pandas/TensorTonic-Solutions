import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    a = np.asarray(a)
    b = np.asarray(b)

    anorm = np.linalg.norm(a)
    bnorm = np.linalg.norm(b)

    if anorm == 0 or bnorm == 0:
        return 0.0

    return float(np.dot(a, b) / (anorm * bnorm))