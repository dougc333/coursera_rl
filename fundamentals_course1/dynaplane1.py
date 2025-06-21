import numpy as np
import matplotlib.pyplot as plt

# Define a simple 1D continuous state space problem
# Objective: approximate value function V(x) using linear planes

# True value function: V(x) = -|x - 2|
def true_value(x):
    return -np.abs(x - 2)

# Initialize state samples
x_vals = np.linspace(0, 4, 100)
V_true = true_value(x_vals)

# Step 1: Sample some points and fit support planes (linear segments)
support_points = np.array([0.0, 1.5, 2.5, 4.0])
V_support = true_value(support_points)

# Fit lines (a_i x + b_i) through adjacent support points
planes = []
for i in range(len(support_points) - 1):
    x1, x2 = support_points[i], support_points[i+1]
    y1, y2 = V_support[i], V_support[i+1]
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    planes.append((a, b))

# Step 2: Define value function approximation using max over planes
def V_approx(x):
    return np.max([a * x + b for a, b in planes], axis=0)

# Evaluate approximation
V_approx_vals = np.array([V_approx(x) for x in x_vals])

# Plot true vs Dynaplane approximation
plt.figure(figsize=(8, 4))
plt.plot(x_vals, V_true, label="True V(x)", linewidth=2)
plt.plot(x_vals, V_approx_vals, label="Dynaplane Approximation", linestyle="--")
plt.scatter(support_points, V_support, c='red', label="Support Points")
plt.title("Dynaplane Value Function Approximation")
plt.xlabel("State x")
plt.ylabel("Value V(x)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
