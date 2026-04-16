import numpy as np
"""
Concept : L'Information Gain (IG) mesure la réduction d'incertitude (entropie) apportée par une division binaire des données sur une feature. Elle calcule la différence entre l'entropie du jeu de données parent et l'entropie pondérée des sous-ensembles gauche/droite après la division.

À quoi ça sert : En apprentissage automatique, c'est un critère clé pour construire des arbres de décision (ex : ID3, C4.5). L'IG permet de choisir la feature qui sépare le mieux les classes, maximisant la pureté des nœuds et améliorant la précision de classification. Plus l'IG est élevé, meilleure est la division. Si IG=0, la division n'apporte aucune information.
"""
def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """
    y = np.asarray(y)
    split_mask = np.asarray(split_mask)
    
    YL = y[split_mask]
    YR = y[~split_mask]
    
    nL = np.sum(split_mask)
    nR = len(y) - nL
    N = len(y)
    
    if nL == 0 or nR == 0:
        return 0.0
    
    H_parent = _entropy(y)
    H_left = _entropy(YL)
    H_right = _entropy(YR)
    
    IG = H_parent - ( (nL / N) * H_left + (nR / N) * H_right )
    return float(IG)