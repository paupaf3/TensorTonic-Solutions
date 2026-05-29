import numpy as np

def binning(values, num_bins):
    """
    Assign each value to an equal-width bin using numpy.
    """
    if not values:
        return []
        
    values_array = np.array(values)
    
    # Check if all values are equal
    if np.min(values_array) == np.max(values_array):
        # Return a list of zeros equal to the length of the input
        return [0] * len(values_array)
        
    # Calculate the bin edges
    _, edges = np.histogram(values_array, bins=num_bins)
    
    # Assign values to bins
    binned_values = np.digitize(values_array, edges[1:-1])
    
    return binned_values.tolist()