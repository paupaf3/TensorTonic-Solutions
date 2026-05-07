import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step, handling both lists and arrays.
    """
    # Convert inputs to numpy arrays to support element-wise math
    w = np.asanyarray(w)
    g = np.asanyarray(g)
    G = np.asanyarray(G)
    
    # 1. Accumulate the squared gradients
    G_new = G + g**2
    
    # 2. Update weights

    w_new = w - (lr / np.sqrt(G_new + eps)) * g
    
    return w_new, G_new