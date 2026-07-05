import math

def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(x1, x2))
    
    # Calculate L2 norms (magnitudes)
    norm_x1 = math.sqrt(sum(a ** 2 for a in x1))
    norm_x2 = math.sqrt(sum(b ** 2 for b in x2))
    
    # Compute cosine similarity safely
    eps = 1e-8
    cos_sim = dot_product / max(norm_x1 * norm_x2, eps)
    
    # Calculate loss based on the label
    if label == 1:
        return 1.0 - cos_sim
    elif label == -1:
        return max(0.0, cos_sim - margin)
    else:
        raise ValueError("Label must be either 1 or -1")