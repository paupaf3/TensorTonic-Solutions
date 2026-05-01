def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    # Convert to a set for O(1)
    stop_set = set(stopwords)
    
    # Filter tokens while maintaining original sequence
    return [token for token in tokens if token not in stop_set]