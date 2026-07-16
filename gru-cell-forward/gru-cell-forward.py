import numpy as np

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    x = np.asarray(x, dtype=float)
    h_prev = np.asarray(h_prev, dtype=float)
    
    D = x.shape[-1]
    H = h_prev.shape[-1]
    
    x_2d, x_was_1d = _as2d(x, D)
    h_prev_2d, h_was_1d = _as2d(h_prev, H)
    
    # Update gate: z_t = sigmoid(x_t @ W_z + h_{t-1} @ U_z + b_z)
    z = _sigmoid(x_2d @ params["Wz"] + h_prev_2d @ params["Uz"] + params["bz"])
    
    # Reset gate: r_t = sigmoid(x_t @ W_r + h_{t-1} @ U_r + b_r)
    r = _sigmoid(x_2d @ params["Wr"] + h_prev_2d @ params["Ur"] + params["br"])
    
    # Candidate hidden state: h_tilde = tanh(x_t @ W_h + (r_t * h_{t-1}) @ U_h + b_h)
    h_tilde = np.tanh(x_2d @ params["Wh"] + (r * h_prev_2d) @ params["Uh"] + params["bh"])
    
    # New hidden state: h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
    h_t = (1 - z) * h_prev_2d + z * h_tilde
    
    if x_was_1d and h_was_1d:
        return h_t[0]
    return h_t