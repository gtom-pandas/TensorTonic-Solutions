import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    word_to_id = {word: i for i, word in enumerate(vocab)}
    vec = np.zeros(len(vocab), dtype=int)

    for token in tokens:
        idx = word_to_id.get(token)
        if idx is not None:
            vec[idx] += 1

    return vec