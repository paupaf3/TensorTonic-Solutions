import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule explicitly.
    """
    # Ensure inputs are numpy arrays
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    
    # Calculate the widths of the trapezoids (change in x)
    dx = np.diff(fpr)
    
    # Calculate the average heights of the trapezoids
    mean_height = (tpr[:-1] + tpr[1:]) / 2.0
    
    # Sum the areas of all trapezoids
    return np.sum(dx * mean_height)