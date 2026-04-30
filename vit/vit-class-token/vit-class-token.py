import numpy as np

def prepend_class_token(patches: np.ndarray, embed_dim: int, cls_token: np.ndarray = None) -> np.ndarray:
    """
    Prepend learnable [CLS] token to patch sequence.
    cls_token: shape (1, 1, D). If None, initialize randomly.
    """
    B, N, D = patches.shape
    
    # 1. Initialize the [CLS] token if not provided
    if cls_token is None:
        cls_token = np.random.randn(1, 1, D) * 0.02
    
    # 2. Tile the [CLS] token across the batch dimension (B, 1, D)
    cls_tokens = np.tile(cls_token, (B, 1, 1))
    return np.concatenate([cls_tokens, patches], axis=1)