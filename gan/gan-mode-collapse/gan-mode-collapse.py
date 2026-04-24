import numpy as np

def detect_mode_collapse(generated_samples, threshold=0.1):
    """
    Returns: dict with "diversity_score" (float) and "is_collapsed" (bool)
    """
    generated_samples = np.asarray(generated_samples)
    diversity_score = np.mean(np.std(generated_samples, axis=0))
    is_collapsed = diversity_score < threshold

    return {
        "diversity_score": round(float(diversity_score), 4),
        "is_collapsed": bool(is_collapsed),
    }