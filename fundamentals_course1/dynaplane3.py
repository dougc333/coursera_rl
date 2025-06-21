import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# Define 2D state space
def true_value(x, y):
    return -torch.abs(x - 2) - torch.abs(y - 2)

# Generate grid
grid_size = 20
x_vals = torch.linspace(0, 4, grid_size)
y_vals = torch.linspace(0, 4, grid_size)
X_grid, Y_grid = torch.meshgrid(x_vals, y_vals, indexing='ij')
Z_true = true_value(X_grid, Y_grid)

# Create state vectors
states = torch.stack([X_grid.flatten(), Y_grid.flatten()], dim=1)

# Initialize V arbitrarily (zeros)
V = torch.zeros(states.shape[0])
gamma = 0.9

# Define simple transition model (stay in place or move slightly)
actions = [torch.tensor([0.0, 0.0]), torch.tensor([0.2, 0.0]), torch.tensor([0.0, 0.2]),
           torch.tensor([-0.2, 0.0]), torch.tensor([0.0, -0.2])]

# Bellman backup loop
for _ in range(10):
    V_new = torch.empty_like(V)
    for i, s in enumerate(states):
        q_vals = []
        for a in actions:
            s_prime = s + a
            s_prime = torch.clamp(s_prime, 0.0, 4.0)
            # Nearest neighbor for s'
            dists = torch.norm(states - s_prime, dim=1)
            j = torch.argmin(dists)
            reward = true_value(*s_prime)
            q_vals.append(reward + gamma * V[j])
        V_new[i] = torch.max(torch.tensor(q_vals))
    V = V_new

# Reshape to grid
Z_approx = V.reshape(grid_size, grid_size).numpy()

# Plot true vs Bellman updated value function
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X_grid.numpy(), Y_grid.numpy(), Z_true.numpy(), cmap='viridis')
ax1.set_title("True Value Function")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("V(x,y)")

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X_grid.numpy(), Y_grid.numpy(), Z_approx, cmap='plasma')
ax2.set_title("Bellman Backup Approximation (PyTorch)")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("V(x,y)")

plt.tight_layout()
plt.show()
