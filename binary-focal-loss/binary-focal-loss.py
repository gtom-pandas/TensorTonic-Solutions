import numpy as np

def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)

    eps = 1e-8
    predictions = np.clip(predictions, eps, 1 - eps)

    pt = np.where(targets == 1, predictions, 1 - predictions)
    loss = -alpha * ((1 - pt) ** gamma) * np.log(pt)

    return float(np.mean(loss))