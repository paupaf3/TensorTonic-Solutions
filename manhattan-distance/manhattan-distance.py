import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    x_0 = np.asarray(x)
    y_0 = np.asarray(y)

    return float(np.sum(np.abs(x_0 - y_0)))