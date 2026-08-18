import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    n = len(seqs)
    l = max_len if max_len is not None else (max(len(seq) for seq in seqs) if n > 0 else 0)
            
    padded = np.full((n, l), fill_value=pad_value)
    
    for i, seq in enumerate(seqs):
        length = min(len(seq), l)
        if length > 0:
            padded[i, :length] = seq[:length]
            
    return padded