import numpy as np
# this is wrong infinite loop 
grid_size = 4
n_states = grid_size * grid_size
n_actions = 4  # up, down, left, right
gamma = 1.0
#           up, down, left, right
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

policy_prob = .25 * np.ones((grid_size,grid_size,4))
print(f"policy_prob:{policy_prob}")

# different init strategies
v_initial = 0 # can be random small values or larger values. SHould reduce convergence time with different values. Faster if closer to 20? 
v = v_initial * np.ones((grid_size,grid_size))

#reward = -1
#blue_reward = -10
# the rewards are below
r = np.ones((grid_size,grid_size)) 
r[0][1] = -10
r[0][2] = -10
r[0][3] = -10
r[2][0] = -10
r[2][1] = -10
r[2][2] = -10
print(f"reward matrix:{r}")

v_prime = np.zeros((grid_size,grid_size))
gamma = 1


# (0,0),(0,1),(0,2),(0,3)
# (1,0),(1,1),(1,2),(1,3)
# (2,0),(2,1),(2,2),(2,3)
# (3,0),(3,1),(3,2),(3,3)



# pos convention, (y, x)
# up is y-1, x
# down is y+1, x 
# right is  y, x+1
# left is y, x-1

# this is wrong, if hits a blue one it returns
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

for i in range(grid_size):
  for j in range(grid_size):
    v[i][j] = r[i][j] + gamma * (policy_prob[i][j][0] * getV_Up((i,j)) + 
                                 policy_prob[i][j][1] * getV_Right((i,j)) + 
                                 policy_prob[i][j][2] * getV_Down((i,j)) + 
                                 policy_prob[i][j][3] * getV_Left((i,j)))

