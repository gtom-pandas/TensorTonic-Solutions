import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.asarray(y)
    
    if len(y) == 0:
        return 0.0
    
    unique, counts = np.unique(y, return_counts=True)
    proportions = counts / len(y)
    
    entropy = 0.0
    for p in proportions:
        if p > 0:
            entropy -= p * np.log2(p)
    
    return float(entropy)
    