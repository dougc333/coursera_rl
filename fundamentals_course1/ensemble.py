import matplotlib.pyplot as plt
import numpy as np

# Simulated reward trends
episodes = np.arange(1, 201)
ensemble_rewards = np.cumsum(np.random.normal(1.5, 0.3, size=200))
single_rewards = np.cumsum(np.random.normal(1.2, 0.4, size=200))

plt.plot(episodes, ensemble_rewards, label='Ensemble Q-Learning')
plt.plot(episodes, single_rewards, label='Single Q-Learning')
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Ensemble vs. Single Q-Learning Performance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()