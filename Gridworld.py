

class Gridworld:
    def __init__(self, size=4):
        self.size = size
        self.grid = [[0 for _ in range(size)] for _ in range(size)]
        self.agent_pos = [0, 0]
        self.goal_pos = [size-1, size-1]

    def get_observation(self):
        return self.grid