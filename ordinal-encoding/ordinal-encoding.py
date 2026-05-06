def ordinal_encoding(values, ordering):
    """
    Encode categorical values using the provided ordering.
    """
    mapping = {ordering[i]: i for i in range(len(ordering))}
    return [mapping[value] for value in values]