import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    seq_len = int(seq_len)
    d_model = int(d_model)
    if seq_len < 1 or d_model < 1:
        raise ValueError("seq_len and d_model must be >= 1")

    pos = np.arange(seq_len, dtype=float)[:, None]  #T, 1

    num_pairs = (d_model + 1) // 2                  # number of sin columns 
    pair_idx = np.arange(num_pairs, dtype=float)[None, :]  # 1, num_pairs

    div = np.power(base, (2.0 * pair_idx) / d_model)       #1, num_pairs
    angles = pos / div                                     # T, num_pairs

    pe = np.empty((seq_len, d_model), dtype=float)
    pe[:, 0::2] = np.sin(angles)                            # even dims: sin

    # odd dims
    if d_model > 1:
        pe[:, 1::2] = np.cos(angles[:, : (d_model // 2)])

    return pe