import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)

    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    B, N_q, d_model = Q.shape
    _, N_k, _ = K.shape
    _, N_v, _ = V.shape

    d_k = d_model // num_heads

    # Linear projections
    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    # Split into heads
    Q_heads = Q_proj.reshape(B, N_q, num_heads, d_k).transpose(0, 2, 1, 3)
    K_heads = K_proj.reshape(B, N_k, num_heads, d_k).transpose(0, 2, 1, 3)
    V_heads = V_proj.reshape(B, N_v, num_heads, d_k).transpose(0, 2, 1, 3)

    # Scaled dot-product attention
    scores = (Q_heads @ K_heads.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    attn = softmax(scores, axis=-1)
    head_out = attn @ V_heads  

    # Concatenate heads: 
    concat = head_out.transpose(0, 2, 1, 3).reshape(B, N_q, d_model)

    # Final output projection
    return concat @ W_o

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    hidden = np.dot(x, W1) + b1
    relu_out = np.maximum(0, hidden)
    return np.dot(relu_out, W2) + b2

def encoder_block(x: np.ndarray,
                  W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray, W_o: np.ndarray,
                  W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray,
                  gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray,
                  num_heads: int) -> np.ndarray:
    """
    Complete Transformer encoder block:
    1) Multi-head attention + residual + layer norm
    2) Feed-forward network + residual + layer norm
    """
    # Self-attention sub-layer
    attn_out = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x1 = layer_norm(x + attn_out, gamma1, beta1)

    # Feed-forward sub-layer
    ff_out = feed_forward(x1, W1, b1, W2, b2)
    out = layer_norm(x1 + ff_out, gamma2, beta2)

    return out