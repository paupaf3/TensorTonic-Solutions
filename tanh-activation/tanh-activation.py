import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x_0 = np.asarray(x)
    return (np.exp(x_0) - np.exp(-x_0)) / (np.exp(x_0) + np.exp(-x_0))