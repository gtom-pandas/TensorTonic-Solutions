import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    # Ensure inputs are numpy arrays
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Compute focal loss terms
    term1 = (1 - p) ** gamma * y * np.log(p)
    term2 = p ** gamma * (1 - y) * np.log(1 - p)
    
    # Full loss (element-wise), then mean
    loss = - (term1 + term2)
    return np.mean(loss)