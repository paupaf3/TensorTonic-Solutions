import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    Standardize X: (X - mean)/std. If 2D and axis=0, per column.
    Return np.ndarray (float).
    """
    x_0= np.asarray(X)
    
    # Calculate mean and standard deviation along the specified axis
    mean = np.mean(x_0, axis=axis, keepdims=True)
    std = np.std(x_0, axis=axis, keepdims=True)
    
    # Subtract mean and divide by std (plus epsilon for stability)
    X_standardized = (x_0 - mean) / (std + eps)
    
    return X_standardized