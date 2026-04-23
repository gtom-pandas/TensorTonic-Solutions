import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)

    if fpr.shape != tpr.shape or fpr.size < 2:
        raise ValueError("fpr and tpr must have the same length and at least 2 points.")

    return float(np.trapezoid(tpr, fpr))