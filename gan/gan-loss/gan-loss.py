import numpy as np

def discriminator_loss(real_probs, fake_probs):
    """Compute discriminator loss using binary cross-entropy."""
    eps = 1e-8
    real_probs = np.clip(np.asarray(real_probs), eps, 1 - eps)
    fake_probs = np.clip(np.asarray(fake_probs), eps, 1 - eps)
    return float(-np.mean(np.log(real_probs) + np.log(1 - fake_probs)))

def generator_loss(fake_probs):
    """Compute non-saturating generator loss."""
    eps = 1e-8
    fake_probs = np.clip(np.asarray(fake_probs), eps, 1 - eps)
    return float(-np.mean(np.log(fake_probs)))