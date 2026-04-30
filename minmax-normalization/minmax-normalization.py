import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0 (default), scale per column.
    Return np.ndarray (float).
    """
    X = np.asarray(X, dtype=float)

    # Find the minimum and maximum values along the specified axis
    x_min = np.min(X, axis=axis, keepdims=True)
    x_max = np.max(X, axis=axis, keepdims=True)

    # Apply the scaling formula
    # Adding eps to the denominator prevents division by zero if max == min
    X_scaled = (X - x_min) / (x_max - x_min + eps)

    return X_scaled