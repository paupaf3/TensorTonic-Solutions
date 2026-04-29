import numpy as np

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    y_true_0 = np.asarray(y_true)
    y_pred_0 = np.asarray(y_pred)

    # In multi-class single-label classification, 
    # micro-F1, micro-precision, and micro-recall are all equal to accuracy.
    tp_total = np.sum(y_true_0 == y_pred_0)
    total_samples = len(y_true_0)

    if total_samples == 0:
        return 0.0

    return float(tp_total / total_samples)