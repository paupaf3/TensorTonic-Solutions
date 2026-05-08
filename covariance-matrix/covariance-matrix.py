import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    Returns None for invalid input.
    """
    # 1. Validate input type and basic structure
    if not isinstance(X, (np.ndarray, list)):
        return None
    
    X = np.asanyarray(X)
    
    # 2. Check dimensions and sample size
    # Covariance requires at least 2 samples for Bessel's correction (n-1)
    if X.ndim != 2 or X.shape[0] < 2:
        return None
        
    # 3. Center the data
    X_centered = X - np.mean(X, axis=0)
    n = X.shape[0]
    
    # 4. Compute the covariance matrix
    cov_mat = (X_centered.T @ X_centered) / (n - 1)
    return cov_mat