import torch
import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, k=10):
        self.k = k
        self.means = torch.randn(k)
        self.stds = torch.rand(k) * 2  # reward std between 0 and 2

    def pull(self, action):
        reward = torch.normal(self.means[action], self.stds[action])
        return reward

class Agent:
    def __init__(self, k, alpha=0.1):
        self.k = k
        self.Q = torch.zeros(k)         # Mean estimate
        self.M2 = torch.zeros(k)        # Second moment estimate
        self.counts = torch.zeros(k)
        self.alpha = alpha              # Step size

    def select_action(self, epsilon=0.1):
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.k, (1,)).item()
        else:
            return torch.argmax(self.Q).item()

    def update(self, action, reward):
        self.counts[action] += 1
        delta = reward - self.Q[action]

        # Update mean
        self.Q[action] += self.alpha * delta

        # Update second moment
        second_term = reward**2 - self.M2[action]
        self.M2[action] += self.alpha * second_term

    def get_variance(self):
        return self.M2 - self.Q**2

# Simulation
k = 10
bandit = Bandit(k)
agent = Agent(k)
steps = 1000

means, variances = [], []

for t in range(steps):
    a = agent.select_action()
    r = bandit.pull(a)
    agent.update(a, r)

    if t % 10 == 0:
        means.append(agent.Q.clone())
        variances.append(agent.get_variance().clone())

# Plot
means = torch.stack(means)
variances = torch.stack(variances)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(means.numpy())
plt.title("Estimated Mean Rewards per Arm")

plt.subplot(1, 2, 2)
plt.plot(variances.numpy())
plt.title("Estimated Variance per Arm")

plt.tight_layout()
plt.show()
