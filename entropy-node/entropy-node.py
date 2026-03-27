import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y_0 = np.asarray(y)

    # Get counts of each class
    unique_values, counts = np.unique(y_0, return_counts=True)
    # Filter classes with 0 counts (errors on log operation)
    counts = counts[counts != 0]
    # Calculate probabilities of each class
    probs = counts / np.sum(counts)

    return -np.sum(probs*np.log2(probs))