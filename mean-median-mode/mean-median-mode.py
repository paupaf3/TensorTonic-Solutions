import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    if not x:
        return None, None, None
        
    # Calculate mean and median using NumPy
    mean_val = np.mean(x)
    median_val = np.median(x)
    
    # Calculate mode using Counter
    # most_common(1) returns a list like [(value, count)], so we extract the value
    mode_val = Counter(x).most_common(1)[0][0]
    
    return mean_val, median_val, mode_val