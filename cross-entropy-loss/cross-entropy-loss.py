import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)
    
    n_samples = len(y_true)
    
    # Extract the predicted probability for the correct class for each smpl
    correct_probs = y_pred[np.arange(n_samples), y_true]
    
    # Compute cross-entropy loss
    loss = -np.mean(np.log(correct_probs))
    
    return float(loss)