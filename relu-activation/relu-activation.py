import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    rel = np.maximum(0,x)
    return rel