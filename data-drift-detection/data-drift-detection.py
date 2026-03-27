import numpy as np

def calculate_tvd(norm_dist_1, norm_dist_2):
    """
    Calculate Total Variation Distance between two normalized distributions
    """
    return 1/2 * np.sum(abs(norm_dist_1 - norm_dist_2))

def detect_drift(reference_counts, production_counts, threshold):
    """
    Compare reference and production distributions to detect data drift.
    """
    reference_counts = np.asarray(reference_counts)
    production_counts = np.asarray(production_counts)
    
    # Normalize 
    reference_norm = reference_counts / np.sum(reference_counts)
    production_norm = production_counts / np.sum(production_counts)
    
    # Calculate score
    tvd = calculate_tvd(reference_norm, production_norm)
    
    return dict(score=tvd, drift_detected=bool(tvd > threshold))