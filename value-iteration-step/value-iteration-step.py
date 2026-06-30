import numpy as np

def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    values = np.asarray(values, dtype=float)
    n_states = len(values)
    updated_values = np.zeros(n_states, dtype=float)

    for s in range(n_states):
        n_actions = len(rewards[s])
        q_values = np.zeros(n_actions, dtype=float)

        for a in range(n_actions):
            reward = rewards[s][a]
            transition_probs = transitions[s][a]
            expected_next_value = np.dot(transition_probs, values)
            q_values[a] = reward + gamma * expected_next_value

        updated_values[s] = np.max(q_values)

    return updated_values.tolist()