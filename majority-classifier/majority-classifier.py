import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    # Stable tie-break: first occurring class in y_train
    labels, first_idx = np.unique(y_train, return_index=True)
    counts = np.array([(y_train == lab).sum() for lab in labels])

    max_count = counts.max()
    tied = np.where(counts == max_count)[0]
    majority_label = labels[tied[np.argmin(first_idx[tied])]]

    return np.full(X_test.shape[0], majority_label, dtype=int)