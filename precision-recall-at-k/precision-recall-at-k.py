
def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Get how many values of K recommended are in the relevant (ground truth)
    hits = len(set(recommended[:k]) & set(relevant))
    # Calculate precision
    precision = hits / k
    # Calculate recall
    recall = hits / len(relevant)

    return [precision, recall]