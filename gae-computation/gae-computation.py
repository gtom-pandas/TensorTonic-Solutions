def gae(rewards, values, gamma, lam):
    """
    Compute Generalized Advantage Estimation.
    """
    T = len(rewards)
    advantages = [0.0] * T
    next_adv = 0.0

    for t in range(T - 1, -1, -1):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        next_adv = delta + gamma * lam * next_adv
        advantages[t] = next_adv

    return advantages