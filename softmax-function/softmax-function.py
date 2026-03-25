import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    x_0 = np.asarray(x)
    if x_0.ndim == 1:
        e_x_0 = np.exp(x_0 - np.max(x_0))
        return e_x_0 / e_x_0.sum()
    else:
        e_x_0 = np.exp(x_0 - np.max(x_0, axis=1, keepdims=True))
        return e_x_0 / e_x_0.sum(axis=1, keepdims=True)