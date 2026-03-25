import numpy as np

def selu(x, lam=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717):
    """
    Apply SELU activation element-wise.
    Returns a list of floats rounded to 4 decimal places.
    """
    x_0 = np.asarray(x)
    return np.where(x_0 > 0, lam * x_0, lam * alpha * (np.exp(x_0) - 1))
