from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow CORS for React frontend


class ParkingWorld:
    def __init__(self,
                 num_spaces=10,
                 num_prices=4,
                 price_factor=0.1,
                 occupants_factor=1.0,
                 null_factor=1 / 3):
        self.__num_spaces = num_spaces
        self.__num_prices = num_prices
        self.__occupants_factor = occupants_factor
        self.__price_factor = price_factor
        self.__null_factor = null_factor
        self.__S = [num_occupied for num_occupied in range(num_spaces + 1)]
        self.__A = list(range(num_prices))


    def transitions(self, s, a):
        return np.array([[r, self.p(s_, r, s, a)] for s_, r in self.support(s, a)])

    def support(self, s, a):
        return [(s_, self.reward(s, s_)) for s_ in self.__S]

    def p(self, s_, r, s, a):
        if r != self.reward(s, s_):
            return 0
        else:
            center = (1 - self.__price_factor
                      ) * s + self.__price_factor * self.__num_spaces * (
                          1 - a / self.__num_prices)
            #print(f"center:{center}")
            
            emphasis = np.exp(
                -abs(np.arange(2 * self.__num_spaces) - center) / 5)
            #print(f"emphasis:{emphasis}")
            if s_ == self.__num_spaces:
                return sum(emphasis[s_:]) / sum(emphasis)
            return emphasis[s_] / sum(emphasis)

    def reward(self, s, s_):
        return self.state_reward(s) + self.state_reward(s_)

    def setTrajectory(self, trajectory):
        self.__trajectory = trajectory
    def getTrajectory(self):
        return self.__trajectory
    def state_reward(self, s):
        if s == self.__num_spaces:
            #print("self_numspaces")
            return self.__null_factor * s * self.__occupants_factor
        else:
            #print("not numspaces")
            return s * self.__occupants_factor

    def random_state(self):
        return np.random.randint(self.__num_prices)

    def step(self, s, a):
        probabilities = [
            self.p(s_, self.reward(s, s_), s, a) for s_ in self.__S
        ]
        print("ParkingWorld step: probabilities:",probabilities, len(probabilities))
        return np.random.choice(self.__S, p=probabilities)


    @property
    def A(self):
        return list(self.__A)

    @property
    def num_spaces(self):
        return self.__num_spaces

    @property
    def num_prices(self):
        return self.num_prices

    @property
    def S(self):
        return list(self.__S)


class Transitions(list):
    def __init__(self, transitions):
        self.__transitions = transitions
        super().__init__(transitions)

    def __repr__(self):
        repr = '{:<14} {:<10} {:<10}'.format('Next State', 'Reward',
                                             'Probability')
        repr += '\n'
        for i, (s, r, p) in enumerate(self.__transitions):
            repr += '{:<14} {:<10} {:<10}'.format(s, round(r, 2), round(p, 2))
            if i != len(self.__transitions) - 1:
                repr += '\n'
        return repr

initial_state = 5
num_steps = 30
env = ParkingWorld(num_spaces=10, num_prices=4)
def init():
    state = initial_state
    trajectory = [state]
    for _ in range(num_steps):
        action = np.random.choice(env.A)  # random pricing action
        # should be greedy action based on max action
        next_state = env.step(state, action)
        trajectory.append(next_state)
        state = next_state
        
    print("trajectory:",trajectory)
    env.setTrajectory(trajectory)
    # Assume `trajectory` is a list or array of state values collected over time
    trajectory_mean = np.mean(trajectory)
    trajectory_variance = np.var(trajectory)

    print("Mean of trajectory:", trajectory_mean)
    print("Variance of trajectory:", trajectory_variance)


GRID_SIZE = 4
agent_pos = [0, 0]
goal_pos = [3, 3]

def get_observation():
    grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    x, y = agent_pos
    gx, gy = goal_pos
    grid[gy][gx] = 2  # goal
    grid[y][x] = 1    # agent (overrides goal if same pos)
    return grid

def step(action):
    global agent_pos
    dx, dy = 0, 0
    if action == "up": dy = -1
    if action == "down": dy = 1
    if action == "left": dx = -1
    if action == "right": dx = 1

    new_x = max(0, min(GRID_SIZE - 1, agent_pos[0] + dx))
    new_y = max(0, min(GRID_SIZE - 1, agent_pos[1] + dy))
    agent_pos = [new_x, new_y]

    reward = 1 if agent_pos == goal_pos else 0
    done = agent_pos == goal_pos
    return [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], reward, done

@app.route('/')
def index():
    return "Hello, World!"

@app.route("/env/reset", methods=["GET"])
def reset():
    global agent_pos
    agent_pos = [0, 0]
    obs = get_observation()[0]
    return [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],1,False

@app.route("/env/step", methods=["POST"])
def env_step():
    action = request.json["action"]
    obs, reward, done = step(action)
    return jsonify({"observation": obs, "reward": reward, "done": done})


@app.route("/trajectory")
def trajectory():
    trajectory = env.getTrajectory()
    app.logger.info("flask trajectory:",trajectory)
    app.logger.info("type trajectory:%s", str(type(trajectory)))
    app.logger.info("len trajectory:%s", str(len(trajectory)))
    data = [int(x) for x in trajectory]
    return jsonify(data)

@app.route("/computetrajectory")
def compute_trajectory():
    state = initial_state
    trajectory = [state]
    for _ in range(num_steps):
        action = np.random.choice(env.A)  # random pricing action
        next_state = env.step(state, action)
        trajectory.append(next_state)
        state = next_state
        
    print("trajectory:",trajectory)
    env.setTrajectory(trajectory)
    # Assume `trajectory` is a list or array of state values collected over time
    trajectory_mean = np.mean(trajectory)
    trajectory_variance = np.var(trajectory)

    print("Mean of trajectory:", trajectory_mean)
    print("Variance of trajectory:", trajectory_variance)


@app.route('/button-click', methods=['GET'])
def handle_button_click():
    #data = request.get_json()
    print(f"/button-click")
    app.logger.info('sending to react component /button-click')
    data = {
        "x": [1, 2, 3, 4, 5],
        "y": [10, 15, 13, 20, 18]
    }
    return jsonify(data)

@app.route('/button-click2', methods=['POST'])
def send_data():    
    data = request.get_json()
    app.logger.info(f"data from react:{data}")
    return jsonify(data)

if __name__ == '__main__':
    init()
    app.run(debug=True, port=5000)
