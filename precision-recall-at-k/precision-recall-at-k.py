def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    recommended_set = set(recommended[:k])
    relevant_set = set(relevant)
    
    hits = len(recommended_set & relevant_set)
    
    precision_at_k = hits / k if k > 0 else 0.0
    recall_at_k = hits / len(relevant_set) if len(relevant_set) > 0 else 0.0
    
    return [precision_at_k, recall_at_k]