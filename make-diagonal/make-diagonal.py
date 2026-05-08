import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal.
    Returns None for invalid input.
    """
    v = np.asanyarray(v)
    
    # Create diagonal matrix
    # np.diag(v) creates an (n, n) matrix with v on the main diagonal
    return np.diag(v)