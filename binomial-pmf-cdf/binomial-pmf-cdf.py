import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    # Ensure k is treated as an integer for the sequence generation
    k = int(k)
    
    # Calculate PMF for exactly k successes
    pmf = comb(n, k) * (p**k) * ((1 - p)**(n - k))
    
    # Calculate CDF for up to k successes
    # Create an array of all possible successes from 0 to k
    i = np.arange(k + 1)
    
    # Calculate the PMF for each value in the array and sum them
    cdf_array = comb(n, i) * (p**i) * ((1 - p)**(n - i))
    cdf = np.sum(cdf_array)
    
    return pmf, cdf