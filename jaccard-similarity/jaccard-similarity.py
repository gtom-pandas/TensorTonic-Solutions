def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    set_a_set = set(set_a)
    set_b_set = set(set_b)
    
    intersection = set_a_set & set_b_set
    union = set_a_set | set_b_set
    
    if len(union) == 0:
        return 0.0
    
    return float(len(intersection)) / len(union)