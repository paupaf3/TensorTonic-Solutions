import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Verify both have the same size
    if len(y_pred) != len(y_true):
        return np.nan
    
    y_pred_0 = np.asarray(y_pred)
    y_true_0 = np.asarray(y_true)

    return (1/y_pred_0.size) * np.sum((y_pred_0 - y_true_0)**2)
