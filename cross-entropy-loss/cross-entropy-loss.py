import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    
    # Clip predictions to avoid log(0) which results in NaN/undefined values
    y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
    
    # If y_true is 1D array of integer labels (e.g., [0, 2, 1])
    if y_true.ndim == 1:
        n_samples = y_true.shape[0]
        correct_confidences = y_pred[np.arange(n_samples), y_true]
        loss = -np.log(correct_confidences)
        return np.mean(loss)
    
    # If y_true is one-hot encoded (e.g., [[1, 0, 0], [0, 0, 1]])
    else:
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]