import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    """
    returns_sum = np.zeros(n_states, dtype=float)
    returns_count = np.zeros(n_states, dtype=int)

    for episode in episodes:
        visited = set()
        T = len(episode)

        for t, (state, _) in enumerate(episode):
            if state in visited:
                continue
            visited.add(state)

            G = 0.0
            discount = 1.0
            for k in range(t, T):
                _, reward = episode[k]
                G += discount * reward
                discount *= gamma

            returns_sum[state] += G
            returns_count[state] += 1

    V = np.zeros(n_states, dtype=float)
    mask = returns_count > 0
    V[mask] = returns_sum[mask] / returns_count[mask]
    return V