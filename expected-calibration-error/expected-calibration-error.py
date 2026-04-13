import numpy as np

def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Compute Expected Calibration Error (ECE) for binary classification.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_bins = int(n_bins)
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not (2 <= n_bins <= 100):
        raise ValueError("n_bins must be between 2 and 100")
    
    n = len(y_true)
    if n == 0:
        return 0.0
    
    # Assign each prediction to a bin
    bin_indices = np.minimum((y_pred * n_bins).astype(int), n_bins - 1)
    
    ece = 0.0
    for m in range(n_bins):
        mask = bin_indices == m
        if not np.any(mask):
            continue 
        
        acc = np.mean(y_true[mask])
        conf = np.mean(y_pred[mask])
        diff = abs(acc - conf)
        weight = np.sum(mask) / n
        ece += weight * diff
    
    return float(ece)