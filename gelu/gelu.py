import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    x_0 = np.asarray(x)
    return (1 / 2) * x_0 * (1 + np.vectorize(math.erf)(x_0/math.sqrt(2)))
