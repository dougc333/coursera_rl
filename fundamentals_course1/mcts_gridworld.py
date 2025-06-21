import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict

# Gridworld setup
grid_size = 4
n_states = grid_size * grid_size
n_actions = 4
#          left     right    down    up
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

# Monte Carlo Tree Search (MCTS) for Gridworld
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0

    def expand(self):
        for a in range(n_actions):
            if a not in self.children:
                next_state, _ = step(self.state, a)
                self.children[a] = MCTSNode(next_state, parent=self)

    def is_fully_expanded(self):
        return len(self.children) == n_actions

    def best_child(self, c_param=1.4):
        choices = [(child.total_reward / (child.visits + 1e-5) + 
                    c_param * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-5)), a, child)
                   for a, child in self.children.items()]
        return max(choices, key=lambda x: x[0])[1:]

    def backpropagate(self, reward):
        self.visits += 1
        self.total_reward += reward
        if self.parent:
            self.parent.backpropagate(reward)

    def tree_policy(self):
        current_node = self
        while not is_terminal(current_node.state):
            current_node.expand()
            if not current_node.is_fully_expanded():
                return current_node
            _, current_node = current_node.best_child()
        return current_node

    def default_policy(self):
        current_state = self.state
        total_reward = 0
        discount = 1.0
        for _ in range(10):
            if is_terminal(current_state):
                break
            a = random.randint(0, n_actions - 1)
            next_state, reward = step(current_state, a)
            total_reward += discount * reward
            discount *= 0.9
            current_state = next_state
        return total_reward

    def best_action(self):
        self.expand()
        for _ in range(100):
            v = self.tree_policy()
            reward = v.default_policy()
            v.backpropagate(reward)
        best_a = max(self.children.items(), key=lambda item: item[1].visits)[0]
        return best_a

# Run MCTS on Gridworld
initial_state = 5
root = MCTSNode(initial_state)
action = root.best_action()
print(f"Best action from state {initial_state} using MCTS: {action}")
