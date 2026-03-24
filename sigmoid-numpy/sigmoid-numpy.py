import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Convert to numpy array and perform calculation
    return 1 / (1 + np.exp(-np.asarray(x, dtype=float)))