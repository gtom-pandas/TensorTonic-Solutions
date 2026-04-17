import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def forget_gate(h_prev: np.ndarray, x_t: np.ndarray,
                W_f: np.ndarray, b_f: np.ndarray) -> np.ndarray:
    """Compute forget gate: f_t = sigmoid(W_f @ [h, x] + b_f)"""
    # Concatenate h_prev and x_t along the last axis =-1
    concat = np.concatenate([h_prev, x_t], axis=-1)
    # Apply linear transformation: concat @ W_f.T + b_f
    linear = concat @ W_f.T + b_f
    # Apply sigmoid activation
    f_t = sigmoid(linear)
    return f_t