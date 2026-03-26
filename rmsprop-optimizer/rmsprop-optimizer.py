import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    w = np.asarray(w)
    g = np.asarray(g)
    s = np.asarray(s)
    
    # STEP 1: Update running average
    s = beta * s + (1 - beta) * g**2
    
    # STEP 2: Parameter update
    w = w - lr / np.sqrt(s + eps) * g

    return (w, s)
    