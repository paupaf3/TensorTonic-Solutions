import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    prediction_error = np.abs(y_true - y_pred)
    return np.mean(np.where(prediction_error <= delta, 0.5 * prediction_error**2, delta*(prediction_error - 0.5*delta)))