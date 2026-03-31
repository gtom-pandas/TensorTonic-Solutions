import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein Critic Loss for WGAN.
    L = E[D(fake)] - E[D(real)]
    """
    real_scores = np.asarray(real_scores)
    fake_scores = np.asarray(fake_scores)
    return float(fake_scores.mean() - real_scores.mean())