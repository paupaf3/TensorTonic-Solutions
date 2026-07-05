def simple_moving_average(values, window_size):
    """
    Compute the simple moving average of the given values.
    """
    if window_size <= 0:
        raise ValueError("Window size must be greater than 0.")
    if window_size > len(values):
        return []

    sma = []
    # Calculate the sum for the very first window
    window_sum = sum(values[:window_size])
    sma.append(window_sum / window_size)

    # Slide the window across the rest of the values
    for i in range(window_size, len(values)):
        window_sum += values[i] - values[i - window_size]
        sma.append(window_sum / window_size)

    return sma