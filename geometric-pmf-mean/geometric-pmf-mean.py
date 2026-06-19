import numpy as np

def geometric_pmf_mean(k, p):
    """
    Compute Geometric PMF and Mean.
    """
    # Convert k to a numpy array to support element-wise operations if a list is passed
    k = np.array(k)
    
    # Calculate the Probability Mass Function (PMF)
    pmf = ((1 - p)**(k - 1)) * p
    
    # Calculate the theoretical mean (Expected Value)
    mean = 1 / p
    
    return pmf, mean