import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    param = np.asarray(param)
    m = np.asarray(m)
    v = np.asarray(v)
    grad = np.asarray(grad)
    
    # Update first moment
    m = beta1 * m + (1 - beta1) * grad
    # Update second moment
    v = beta2 * v + (1 - beta2) * grad**2

    # Bias correction only to update parameter
    m_bias = m / (1 - beta1**t)
    v_bias = v / (1 - beta2**t)
    
    # Update parameter
    param = param - lr * m_bias / (np.sqrt(v_bias) + eps)
    
    return (param, m, v)