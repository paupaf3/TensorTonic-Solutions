import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    x_0 = np.asarray(x)
    y_0 = np.asarray(y)

    return float(np.sqrt(np.sum((x_0 - y_0)**2)))