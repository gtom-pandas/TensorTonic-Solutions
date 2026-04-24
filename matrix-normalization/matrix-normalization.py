import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    try:
        matrix = np.asarray(matrix, dtype=float)

        if matrix.ndim != 2:
            return None

        if axis not in (0, 1, None):
            return None

        if norm_type == 'l2':
            denom = np.sqrt(np.sum(matrix ** 2, axis=axis, keepdims=True))
        elif norm_type == 'l1':
            denom = np.sum(np.abs(matrix), axis=axis, keepdims=True)
        elif norm_type == 'max':
            denom = np.max(np.abs(matrix), axis=axis, keepdims=True)
        else:
            return None

        denom = np.where(denom == 0, 1, denom)
        return matrix / denom

    except Exception:
        return None