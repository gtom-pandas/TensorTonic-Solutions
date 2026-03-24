import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    B, N_q, d_model = Q.shape
    Bk, N_k, d_model_k = K.shape
    Bv, N_v, d_model_v = V.shape

    assert B == Bk == Bv
    assert d_model == d_model_k == d_model_v
    assert d_model % num_heads == 0

    d_k = d_model // num_heads

    # Linear projections
    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    # Split into heads
    Q_heads = Q_proj.reshape(B, N_q, num_heads, d_k).transpose(0, 2, 1, 3)
    K_heads = K_proj.reshape(B, N_k, num_heads, d_k).transpose(0, 2, 1, 3)
    V_heads = V_proj.reshape(B, N_v, num_heads, d_k).transpose(0, 2, 1, 3)

    # Scaled dot-product attention per head
    scores = (Q_heads @ K_heads.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    attn = softmax(scores, axis=-1)
    head_outputs = attn @ V_heads 

    # Concatenate heads
    concat = head_outputs.transpose(0, 2, 1, 3).reshape(B, N_q, d_model)

    # Output projection
    output = concat @ W_o

    return output