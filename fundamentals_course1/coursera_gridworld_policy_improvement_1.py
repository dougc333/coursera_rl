import numpy as np
# this is wrong infinite loop 
grid_size = 4
n_states = grid_size * grid_size
n_actions = 4  # up, down, left, right
gamma = 1.0

actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

policy_prob = .25 * np.ones((grid_size,grid_size,4))
print(policy_prob)

# different init strategies
v_initial = 0 # can be random small values or larger values. SHould reduce convergence time with different values. Faster if closer to 20? 
v = v_initial * np.ones((grid_size,grid_size))
v_prime = np.ones((grid_size,grid_size))
reward = -1
gamma = 1

policy_per_cell:dict[tuple, list]={}
policy_per_cell[(0,0)] = ["left", "right"]
print(policy_per_cell)
for i in range(grid_size):
  for j in range(grid_size):
    policy_per_cell[(i,j)] = []

print(policy_per_cell)



# pos convention, (y, x)
# up is y-1, x
# down is y+1, x 
# right is  y, x+1
# left is y, x-1

# (0,0),(0,1),(0,2),(0,3)
# (1,0),(1,1),(1,2),(1,3)
# (2,0),(2,1),(2,2),(2,3)
# (3,0),(3,1),(3,2),(3,3)

def goUp(pos:tuple)-> tuple:
  if pos[0] - 1 < 0:
    return (0,pos[1])
  return pos[0]-1, pos[1]

def goRight(pos:tuple)-> tuple:
  if pos[1] + 1 > grid_size -1: 
    return (pos[0], grid_size - 1)
  return (pos[0], pos[1] + 1) 

def goDown(pos:tuple)-> tuple:
  if pos[0] + 1 > grid_size -1:
    return (grid_size - 1, pos[1])
  return (pos[0] + 1, pos[1])

def goLeft(pos:tuple)-> tuple:
  if pos[1] -1 < 0:
    return (pos[0], 0)
  return (pos[0], pos[1]-1)

print(f"pos:{(0,0)} test goUp:{goUp((0,0))}")
assert((0,0)==goUp((0,0)))
print(f"pos:{(2,2)} test goUp:{goUp((2,2))} ")
assert((1,2)==goUp((2,2)))
print(f"pos:{(3,3)} test goRight:{goRight((3,3))} ")
assert(((3,3))==goRight((3,3)))
print(f"pos:{(2,3)} test goRight:{goRight((2,3))} ")
assert( (2,3)==goRight((2,3)) ) 
print(f"pos:{(3,1)} test goDown:{goDown((3,1))} ")
assert( (3,1) == goDown((3,1)))
print(f"pos:{(1,1)} test goDown:{goDown((1,1))} ")
assert((2,1)==goDown((1,1)))
print(f"pos:{(2,0)} test goLeft:{goLeft((2,0))} ")
assert((2,0)==goLeft((2,0)))
print(f"pos:{(2,1)} test goLeft:{goLeft((2,1))} ")
assert((2,0)==goLeft((2,1)))

terminal_states = [(0,0),(3,3)]
def isTerminal(pos)->bool:
  if pos in terminal_states:
    return True
  return False

def getV_Up(pos:tuple)->float:
  upPos = goUp(pos)
  return v[upPos[0], upPos[1]]

def getV_Right(pos:tuple)->float:
  rightPos = goRight(pos)
  return v[rightPos[0]][rightPos[1]]

def getV_Down(pos:tuple)->float:
  downPos = goDown(pos)
  return v[downPos[0]][downPos[1]]

def getV_Left (pos:tuple)->float:
  leftPos = goLeft(pos)
  return v[leftPos[0]][leftPos[1]]

theta = 0.01
delta = 1000000
#delta = 0.
iter = 0
while delta > theta:
  delta = 0.
  for i in range(grid_size):
    for j in range(grid_size):
      if isTerminal((i,j)):
        v_prime[i][j] = 0.
      else:
        # go through actions and calculate new v
        v_prime[i][j] = \
        .25*(reward + gamma * getV_Up((i,j))) \
        +.25*(reward + gamma * getV_Right((i,j))) \
        +.25*(reward + gamma * getV_Down((i,j))) \
        +.25*(reward + gamma * getV_Left((i,j))) 
    diff = np.abs(v-v_prime)
    max_diff = np.max(diff)
    min_diff = np.min(diff)
    delta = max(delta, np.abs(v[i,j]-v_prime[i,j]))
    #print(f"diff:{diff}")
    #print(f"min_diff:{min_diff} max_diff:{max_diff} element_diff:{np.abs(v[i,j]-v_prime[i,j])}")
    #print(f"delta:{delta} theta:{theta} iter:{iter}")
  iter +=1
  v = v_prime.copy()
   
print("v_prime:",v_prime)
print("*"*30)
print("find best actions")
best_actions:dict[tuple, list] = {}
for i in range(grid_size):
  for j in range(grid_size):
    print("-"*20)
    print(f"v[{i}][{j}]")
    arr=[getV_Up((i,j)),getV_Right((i,j)),getV_Down((i,j)),getV_Left((i,j))]
    #print(f"arr:{arr} max:{np.max(arr)}")
    max_indexes = np.where(arr == np.max(arr))
    #print(f"max_indexes:{max_indexes}")
    if 0 in max_indexes[0]:
      print("up")
      try:
        best_actions[(i,j)].append("up")
      except KeyError:
        best_actions[(i,j)]=[]
        best_actions[(i,j)].append("up")
    if 1 in max_indexes[0]:
      print("right")
      try:
        best_actions[(i,j)].append("right")
      except KeyError:
        best_actions[(i,j)]=[]
        best_actions[(i,j)].append("right")
    if 2 in max_indexes[0]:
      print("down")
      try:
        best_actions[(i,j)].append("down")
      except KeyError:
        best_actions[(i,j)]=[]
        best_actions[(i,j)].append("down")
    if 3 in max_indexes[0]:
      print("left")
      try:
        best_actions[(i,j)].append("left")
      except KeyError:
        best_actions[(i,j)]=[]
        best_actions[(i,j)].append("left")
print("\n")
print(f"best_actions:{best_actions}")
print("\n")
