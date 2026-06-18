import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    Standardize X: (X - mean)/std. If 2D and axis=0, per column.
    Return np.ndarray (float).
    """
 
    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True)
    
    # Avoid division by zero 
    std_safe = std + eps
    
    # Standardize the data
    standardized_X = (X - mean) / std_safe
    
    return standardized_X