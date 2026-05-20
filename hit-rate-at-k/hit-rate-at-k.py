def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    if not recommendations:
        return 0.0

    hits = 0

    for recs, truth in zip(recommendations, ground_truth):
        top_k = recs[:k]
        if any(item in truth for item in top_k):
            hits += 1

    return hits / len(recommendations)