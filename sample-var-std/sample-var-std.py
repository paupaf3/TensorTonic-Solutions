import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    # Calculate sample variance with degrees of freedom = 1
    variance = np.var(x, ddof=1)
    
    # Calculate sample standard deviation with degrees of freedom = 1
    std_dev = np.std(x, ddof=1)
    
    return variance, std_dev