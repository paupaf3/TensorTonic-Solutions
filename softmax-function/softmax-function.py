import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Subtract the maximum value along the last axis for numerical stability
    # keepdims=True ensures proper broadcasting for both 1D and 2D arrays
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    
    # Divide by the sum of exponents along the last axis
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)