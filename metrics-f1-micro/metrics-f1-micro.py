def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true != pred)
    fn = fp
    
    if 2 * tp + fp + fn == 0:
        return 0.0
    
    f1 = (2 * tp) / (2 * tp + fp + fn)
    return float(f1)