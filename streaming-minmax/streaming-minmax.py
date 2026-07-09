import numpy as np

def streaming_minmax_init(D):
    """
    Initialize state dict with min, max arrays of shape (D,).
    """
    return {
        'min': np.full(D, np.inf, dtype=float),
        'max': np.full(D, -np.inf, dtype=float)
    }

def streaming_minmax_update(state, X_batch, eps=1e-8):
    """
    Update state's min/max with X_batch, return normalized batch.
    """
    X_batch = np.asarray(X_batch, dtype=float)
    
    # Update running min and max
    state['min'] = np.minimum(state['min'], X_batch.min(axis=0))
    state['max'] = np.maximum(state['max'], X_batch.max(axis=0))
    
    # Normalize batch
    denominator = state['max'] - state['min'] + eps
    X_normalized = (X_batch - state['min']) / denominator
    
    return X_normalized