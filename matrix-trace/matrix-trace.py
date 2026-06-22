import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements) explicitly.
    """
    # Ensure input is a numpy array
    A = np.array(A)
    
    # Extract the main diagonal and calculate its sum
    return np.sum(np.diagonal(A))