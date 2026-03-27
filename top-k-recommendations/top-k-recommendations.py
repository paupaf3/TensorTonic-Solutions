def top_k_recommendations(scores, rated_indices, k):
    """
    Return indices of top-k unrated items by predicted score.
    """
    # STEP 1 Filter out items that appear in the rated set
    unrated = [(s, i) for i, s in enumerate(scores) if i not in rated_indices]   
    # STEP 2 Sort the reamining items by their predicted 
    # score in descending order and return top k INDICES
    ans = [i for s, i in sorted(unrated, key=lambda x: (x[0], -x[1]), reverse=True)]
    return ans[:k]