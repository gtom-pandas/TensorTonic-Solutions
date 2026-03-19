import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    Returns a matrix of shape (seq_length, d_model).
    """
    # Positions: shape (seq_length, 1)
    positions = np.arange(seq_length).reshape(-1, 1)

    # Dimension indices for the sine/cosine pairs: shape (d_model/2,)
    div_term = np.exp(
        np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
    )

    # PE matrix
    pe = np.zeros((seq_length, d_model))

    # Apply sine to even indices
    pe[:, 0::2] = np.sin(positions * div_term)

    # Apply cosine to odd indices
    pe[:, 1::2] = np.cos(positions * div_term)

    return pe