import numpy as np

def build_swf_transition_matrix(grid_size=4, slip_prob=0.2):
    actions = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1)    # right
    }
    n_states = grid_size * grid_size
    n_actions = 4

    # Initialize transition probability matrix
    P = np.zeros((n_states, n_actions, n_states))

    def state_to_pos(s):
        return divmod(s, grid_size)

    def pos_to_state(pos):
        return pos[0] * grid_size + pos[1]

    for s in range(n_states):
        row, col = state_to_pos(s)
        for a in range(n_actions):
            probs = np.zeros(n_states)

            # Primary action
            intended_move = actions[a]
            next_row = np.clip(row + intended_move[0], 0, grid_size - 1)
            next_col = np.clip(col + intended_move[1], 0, grid_size - 1)
            next_s = pos_to_state((next_row, next_col))
            probs[next_s] += 1 - slip_prob

            # Slips to other directions
            other_actions = [k for k in actions if k != a]
            for alt_a in other_actions:
                alt_move = actions[alt_a]
                alt_row = np.clip(row + alt_move[0], 0, grid_size - 1)
                alt_col = np.clip(col + alt_move[1], 0, grid_size - 1)
                alt_s = pos_to_state((alt_row, alt_col))
                probs[alt_s] += slip_prob / 3

            # Assign transition probabilities
            P[s, a, :] = probs

    return P

# Example usage
P = build_swf_transition_matrix()
print("Shape of transition matrix:", P.shape)
print("Example state 5, action 2 (left) transitions:")
print(P[5, 2])
