import numpy as np

def ndcg(relevance_scores, k):
    rel = np.asarray(relevance_scores, dtype=np.float64)
    if rel.size == 0:
        return 0.0

    k = int(k)
    if k <= 0:
        raise ValueError("k must be a positive integer")
    kk = min(k, rel.size)

    def dcg(vals):
        gains = np.power(2.0, vals) - 1.0
        pos = np.arange(1, vals.size + 1, dtype=np.float64) 
        discounts = np.log2(pos + 1.0)
        return float(np.sum(gains / discounts))

    dcg_k = dcg(rel[:kk])
    idcg_k = dcg(np.sort(rel)[::-1][:kk])

    return 0.0 if idcg_k == 0.0 else float(dcg_k / idcg_k)