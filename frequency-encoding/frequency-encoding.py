def frequency_encoding(values):
    """
    Replace each value with its frequency proportion.
    """
    counts = {value: values.count(value) for value in values}
    return  [counts[value] / len(values) for value in values]