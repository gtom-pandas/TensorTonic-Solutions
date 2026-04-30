import numpy as np

def add_position_embedding(patches: np.ndarray,num_patches: int,embed_dim: int,pos_embed: np.ndarray = None,) -> np.ndarray:
    """
    Add learnable position embeddings to patch embeddings.
    """
    if not isinstance(patches, np.ndarray):
        raise TypeError("patches must be a numpy.ndarray")

    if patches.ndim != 3:
        raise ValueError(f"patches must have 3 dimensions (B, N, D), got shape {patches.shape}")

    B, N, D = patches.shape

    if N != num_patches:
        raise ValueError(f"num_patches ({num_patches}) does not match patches.shape[1] ({N})")
    if D != embed_dim:
        raise ValueError(f"embed_dim ({embed_dim}) does not match patches.shape[2] ({D})")

    if pos_embed is None:
        pos_embed = np.random.randn(1, num_patches, embed_dim) * 0.02
    else:
        pos_embed = np.asarray(pos_embed)
        # Accept (N, D) or (1, N, D)
        if pos_embed.shape == (num_patches, embed_dim):
            pos_embed = pos_embed.reshape(1, num_patches, embed_dim)
        if pos_embed.shape != (1, num_patches, embed_dim):
            raise ValueError(
                "pos_embed must have shape (1, N, D) or (N, D); "
                f"got {pos_embed.shape}"
            )

    # Element-wise addition
    return patches + pos_embed