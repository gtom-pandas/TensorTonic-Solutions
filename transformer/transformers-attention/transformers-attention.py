import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    d_k = Q.shape[-1]

    # Attention scores: [batch, seq_len_q, seq_len_k]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Normalize scores into attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # Weighted sum of values: [batch, seq_len_q, d_v]
    output = torch.matmul(attention_weights, V)

    return output