import numpy as np
import matplotlib.pyplot as plt

# Gridworld parameters
grid_size = 4
n_states = grid_size * grid_size
n_actions = 4  # up, down, left, right
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def is_terminal(s):
    return s == 0 or s == n_states - 1

def step(s, a):
    row, col = divmod(s, grid_size)
    dr, dc = actions[a]
    new_row = np.clip(row + dr, 0, grid_size - 1)
    new_col = np.clip(col + dc, 0, grid_size - 1)
    s_prime = new_row * grid_size + new_col
    reward = 0 if is_terminal(s) else -1
    return s_prime, reward

# Value Iteration
def value_iteration(gamma=1.0, theta=1e-4):
    V = np.zeros(n_states)
    while True:
        delta = 0
        for s in range(n_states):
            if is_terminal(s):
                continue
            q_vals = []
            for a in range(n_actions):
                s_prime, reward = step(s, a)
                q_vals.append(reward + gamma * V[s_prime])
            max_q = max(q_vals)
            delta = max(delta, abs(max_q - V[s]))
            V[s] = max_q
        if delta < theta:
            break
    # Extract policy
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        q_vals = [step(s, a)[1] + gamma * V[step(s, a)[0]] for a in range(n_actions)]
        policy[s] = np.argmax(q_vals)
    return V, policy

# Policy Iteration
def policy_iteration(gamma=1.0):
    policy = np.zeros(n_states, dtype=int)
    V = np.zeros(n_states)
    while True:
        # Policy Evaluation
        while True:
            delta = 0
            for s in range(n_states):
                if is_terminal(s):
                    continue
                a = policy[s]
                s_prime, reward = step(s, a)
                v = reward + gamma * V[s_prime]
                delta = max(delta, abs(v - V[s]))
                V[s] = v
            if delta < 1e-4:
                break
        # Policy Improvement
        policy_stable = True
        for s in range(n_states):
            old_action = policy[s]
            q_vals = [step(s, a)[1] + gamma * V[step(s, a)[0]] for a in range(n_actions)]
            policy[s] = np.argmax(q_vals)
            if old_action != policy[s]:
                policy_stable = False
        if policy_stable:
            break
    return V, policy

# Run both algorithms
V_val, pi_val = value_iteration()
V_pol, pi_pol = policy_iteration()

# Compare visually
def plot_values(values, title):
    plt.imshow(values.reshape((grid_size, grid_size)), cmap='coolwarm', origin='upper')
    plt.colorbar()
    plt.title(title)
    plt.xticks([])
    plt.yticks([])

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plot_values(V_val, "Value Iteration V")
plt.subplot(1, 2, 2)
plot_values(V_pol, "Policy Iteration V")
plt.tight_layout()
plt.show()
