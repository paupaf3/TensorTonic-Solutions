import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    x = np.sort(np.asarray(x))
    return np.percentile(x, q, method='linear')