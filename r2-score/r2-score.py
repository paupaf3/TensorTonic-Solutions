import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Edge cases
    if (y_true == y_true[0]).all():
        return float((y_true == y_pred).all())

    y_true_mean = np.mean(y_true, keepdims=True)
    
    return 1 - np.sum((y_pred - y_true)**2) / np.sum((y_true_mean - y_true)**2)