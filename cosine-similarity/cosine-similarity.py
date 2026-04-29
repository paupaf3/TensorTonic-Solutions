import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    a_0 = np.asarray(a)
    b_0 = np.asarray(b)

    # Calculate the dot product (numerator)
    dot_product = np.dot(a_0, b_0)
    
    # Calculate the product of the magnitudes (denominator)
    norm_a = np.linalg.norm(a_0)
    norm_b = np.linalg.norm(b_0)
    
    # Handle the case where one vector is a zero vector to avoid division by zero
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))