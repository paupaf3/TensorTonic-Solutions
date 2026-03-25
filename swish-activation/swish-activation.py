import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    x_0 = np.asarray(x)
    return x_0 * (1 / (1 + np.exp(-x_0)))