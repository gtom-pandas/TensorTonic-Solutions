import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int, W_proj: np.ndarray = None) -> np.ndarray:
    """
    Convert image to patch embeddings.
    W_proj: projection matrix of shape (patch_dim, embed_dim). If None, initialize randomly.
    """
    B, H, W, C = image.shape
    N = (H // patch_size) * (W // patch_size)
    patch_dim = patch_size * patch_size * C

    if W_proj is None:
        W_proj = np.random.randn(patch_dim, embed_dim) * 0.02

    patches = image.reshape(
        B,
        H // patch_size,
        patch_size,
        W // patch_size,
        patch_size,
        C
    )
    patches = patches.transpose(0, 1, 3, 2, 4, 5)
    patches = patches.reshape(B, N, patch_dim)

    embeddings = patches @ W_proj
    return embeddings