import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x_0 = np.asarray(x)
    p_0 = np.asarray(p)

    # If probabilities don't sum to 1
    if np.sum(p) != 1:
        raise ValueError
        
    return np.sum(x_0 * p)
