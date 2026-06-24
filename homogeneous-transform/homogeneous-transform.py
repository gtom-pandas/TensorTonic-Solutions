import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    T = np.asarray(T, dtype=float)
    points = np.asarray(points, dtype=float)

    single_point = points.ndim == 1
    if single_point:
        points = points[None, :]

    ones = np.ones((points.shape[0], 1), dtype=float)
    points_h = np.hstack([points, ones])

    transformed_h = (T @ points_h.T).T
    transformed = transformed_h[:, :3]

    return transformed[0] if single_point else transformed