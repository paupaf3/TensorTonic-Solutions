import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    x_0 = np.asarray(x)
    return np.where(x_0 >= 0, x_0, alpha*x_0)