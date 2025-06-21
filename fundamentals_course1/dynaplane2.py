import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Define a 2D continuous state space
# True value function: V(x, y) = -|x - 2| - |y - 2|
def true_value(x, y):
    return -np.abs(x - 2) - np.abs(y - 2)

# Generate sample points in 2D
x_vals = np.linspace(0, 4, 20)
y_vals = np.linspace(0, 4, 20)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
Z_true = true_value(X_grid, Y_grid)

# Sample support points randomly for regression
np.random.seed(0)
support_points = np.random.uniform(0, 4, size=(50, 2))
V_support = true_value(support_points[:, 0], support_points[:, 1])

# Fit linear regression models to local regions (Dynaplane planes)
# We'll divide the support points into clusters and fit a plane to each
from sklearn.cluster import KMeans
k = 5
kmeans = KMeans(n_clusters=k, random_state=0).fit(support_points)

planes = []
for i in range(k):
    cluster_points = support_points[kmeans.labels_ == i]
    cluster_values = V_support[kmeans.labels_ == i]
    if len(cluster_points) > 2:
        model = LinearRegression().fit(cluster_points, cluster_values)
        coef = model.coef_
        intercept = model.intercept_
        planes.append((coef, intercept))

# Value function approximation as max over planes
def V_approx(x, y):
    inputs = np.stack([x, y], axis=-1)
    results = np.array([np.dot(inputs, coef) + b for coef, b in planes])
    return np.max(results, axis=0)

# Evaluate approximation on grid
Z_approx = V_approx(X_grid, Y_grid)

# Plot true vs approximate value function
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X_grid, Y_grid, Z_true, cmap='viridis')
ax1.set_title("True Value Function")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("V(x,y)")

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X_grid, Y_grid, Z_approx, cmap='plasma')
ax2.set_title("Dynaplane Approximation")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("V(x,y)")

plt.tight_layout()
plt.show()
