import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Calculate confusion matrix with normalization options.
    """
    # Convert inputs to numpy arrays for vectorized operations
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Check that y_true and y_pred have the same length
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length :)")
    
    # Handle the edge case of empty arrays
    if len(y_true) == 0:
        if num_classes is None:
            raise ValueError("num_classes must be specified for empty arrays")
        # Return a zero matrix of the appropriate type and shape
        dtype = np.int64 if normalize == 'none' else np.float64
        return np.zeros((num_classes, num_classes), dtype=dtype)
    
    # Infer num_classes if not provided, as max of labels + 1
    if num_classes is None:
        num_classes = int(max(np.max(y_true), np.max(y_pred)) + 1)
    
    # Validate that num_classes is a positive integer
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")
    
    # Validate that all labels are in the valid range [0, num_classes)
    if np.any(y_true < 0) or np.any(y_true >= num_classes) or np.any(y_pred < 0) or np.any(y_pred >= num_classes):
        raise ValueError("All labels must be in range [0, num_classes)")
    
    # Validate the normalize parameter
    if normalize not in ['none', 'true', 'pred', 'all']:
        raise ValueError("normalize must be one of 'none', 'true', 'pred', 'all'")
    
    # Compute indices for bincount: each index is true_label * K + pred_label
    indices = y_true.astype(np.int64) * num_classes + y_pred.astype(np.int64)
    # Use bincount to count occurrences of each (true, pred) pair
    conf_flat = np.bincount(indices, minlength=num_classes**2)
    # Reshape the flat array into a KxK matrix
    cm = conf_flat.reshape(num_classes, num_classes).astype(np.float64)
    
    # Apply row-wise normalization (normalize='true')
    if normalize == 'true':
        row_sums = cm.sum(axis=1, keepdims=True)
        # Set zero row sums to 1 to avoid division by zero
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm /= row_sums
    # Apply column-wise normalization (normalize='pred')
    elif normalize == 'pred':
        col_sums = cm.sum(axis=0, keepdims=True)
        # Set zero column sums to 1 to avoid division by zero
        col_sums = np.where(col_sums == 0, 1, col_sums)
        cm /= col_sums
    # Apply total normalization (normalize='all')
    elif normalize == 'all':
        total = cm.sum()
        if total > 0:
            cm /= total
    
    # Return the matrix as int64 for 'none', float64 otherwise
    if normalize == 'none':
        return cm.astype(np.int64)
    else:
        return cm

"""
This one was quite hard iterate multiple times to achieve this problem 
"""