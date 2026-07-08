import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Get the number of samples (N) and features (D)
    N, D = X.shape
    
    # Initialize weights and bias to zeros
    w = np.zeros(D)
    b = 0.0
    
    # Ensure y is a 1D array to prevent broadcasting dimension mismatches
    y = np.squeeze(y)
    
    for _ in range(steps):
        # Forward pass: compute the linear combination and apply the activation
        z = np.dot(X, w) + b
        A = _sigmoid(z)
        
        # Compute the gradients of the loss with respect to w and b
        dz = A - y
        dw = (1 / N) * np.dot(X.T, dz)
        db = (1 / N) * np.sum(dz)
        
        # Update the parameters
        w -= lr * dw
        b -= lr * db
        
    return w, b