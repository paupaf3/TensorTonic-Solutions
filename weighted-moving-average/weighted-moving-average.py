def weighted_moving_average(values, weights):
    """
    Compute the weighted moving average using the given weights.
    
    Args:
        values (list of int/float): The data points to average.
        weights (list of int/float): The weights to apply to each window.
        
    Returns:
        list of float: The weighted moving averages.
    """
    if not values or not weights:
        return []

    window_size = len(weights)
    if len(values) < window_size:
        return []

    weight_sum = sum(weights)
    if weight_sum == 0:
        raise ValueError("Sum of weights cannot be zero to avoid division by zero.")

    wma = []
    # Slide the window across the values array
    for i in range(len(values) - window_size + 1):
        window = values[i : i + window_size]
        # Calculate the dot product of the window and weights
        weighted_sum = sum(v * w for v, w in zip(window, weights))
        wma.append(weighted_sum / weight_sum)

    return wma