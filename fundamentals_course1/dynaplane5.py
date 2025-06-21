import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# Define 2D state space and belief space
def true_value(x, y):
    return -torch.abs(x - 2) - torch.abs(y - 2)

# Grid setup
grid_size = 20
x_vals = torch.linspace(0, 4, grid_size)
y_vals = torch.linspace(0, 4, grid_size)
X_grid, Y_grid = torch.meshgrid(x_vals, y_vals, indexing='ij')
states = torch.stack([X_grid.flatten(), Y_grid.flatten()], dim=1)
n_states = states.shape[0]

# Belief: probability distribution over states
beliefs = torch.eye(n_states)  # initial beliefs: 1-hot per state for simplicity

# Transition model (identity with small random noise)
actions = [torch.tensor([0.0, 0.0]), torch.tensor([0.2, 0.0]), torch.tensor([0.0, 0.2]),
           torch.tensor([-0.2, 0.0]), torch.tensor([0.0, -0.2])]

def transition(s, a):
    s_prime = s + a + 0.05 * torch.randn_like(s)
    return torch.clamp(s_prime, 0.0, 4.0)

def observe(s):
    noise = 0.1 * torch.randn_like(s)
    return torch.clamp(s + noise, 0.0, 4.0)

def observation_prob(o, s):
    return torch.exp(-torch.norm(o - s) ** 2 / 0.1)  # Gaussian-like

# Initialize value function over beliefs
V = torch.zeros(n_states)
gamma = 0.9
belief_evolution = []  # track evolving beliefs over time

# POMDP belief backup loop
for step in range(5):
    V_new = torch.empty_like(V)
    belief_snapshot = torch.zeros_like(beliefs)
    for i in range(n_states):
        b = beliefs[i]  # belief over states
        q_vals = []
        best_b_prime = None
        best_q_val = -float('inf')
        for a in actions:
            b_prime = torch.zeros_like(b)
            for j in range(n_states):
                s = states[j]
                s_prime = transition(s, a)
                o = observe(s_prime)
                weights = torch.tensor([observation_prob(o, s_k) for s_k in states])
                weights = weights / weights.sum()
                b_prime += b[j] * weights
            b_prime = b_prime / b_prime.sum()
            expected_value = torch.dot(b_prime, V)
            immediate_reward = torch.dot(b, torch.tensor([true_value(s[0], s[1]) for s in states]))
            q_val = immediate_reward + gamma * expected_value
            q_vals.append(q_val)
            if q_val > best_q_val:
                best_q_val = q_val
                best_b_prime = b_prime
        V_new[i] = torch.max(torch.stack(q_vals))
        belief_snapshot[i] = best_b_prime
    V = V_new
    beliefs = belief_snapshot
    belief_evolution.append(belief_snapshot.mean(dim=0))

# Plot true value function
Z_true = true_value(X_grid, Y_grid).numpy()
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X_grid.numpy(), Y_grid.numpy(), Z_true, cmap='viridis')
ax1.set_title("True Value Function")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("V(x,y)")

# Plot belief evolution heatmaps
fig2, axes = plt.subplots(1, len(belief_evolution), figsize=(15, 3))
for idx, b in enumerate(belief_evolution):
    heatmap = b.reshape(grid_size, grid_size).numpy()
    ax = axes[idx]
    im = ax.imshow(heatmap, origin='lower', cmap='hot', extent=[0, 4, 0, 4])
    ax.set_title(f"Belief t={idx+1}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
fig2.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
plt.tight_layout()
plt.show()