import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    new_G = G + np.square(g)
    new_w = w - lr/np.sqrt(new_G + eps)*g
    return new_w , new_G