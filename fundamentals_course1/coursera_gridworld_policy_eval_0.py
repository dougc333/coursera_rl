import numpy as np


grid_size = 4

# uniform random policy hardcoded as .25 in v_prime calculation also means each direction equiprobable

# (0,0), (0,1),(0,2),(0,3)
# (1,0), (1,1), (1,2), (1,3)
# (2,0), (2,1), (2,2), (2,3)
# (3,0), (3,1), (3,2), (3,3)

actions = {
        "up": (-1, 0),  # up 
        "down": (1, 0),   # down 
        "left": (0, -1),  # left 
        "right": (0, 1)    # right 
}
n_states = grid_size * grid_size
n_actions = 4
v = np.zeros((grid_size, grid_size))
v_prime = np.zeros((grid_size, grid_size))

P = np.zeros((n_states, n_actions, n_states))

def state_to_pos(s):
  return divmod(s, grid_size)

def pos_to_state(pos):
  return pos[0] * grid_size + pos[1]


def test_pos_to_states()->int:
  for i in range(grid_size):
    for j in range(grid_size):
      #print(i,j)
      print("state:",(i,j), "state: ",pos_to_state((i,j)))

#test_pos_to_states()

def test():
  print("test do you see a nicely formatted 4x4 matrix with zeros?")
  print(v)

  print("test state_to_pos")
  print("state0 at grid coords:", state_to_pos(0))
  print("state4 at grid coords:", state_to_pos(4))
  print("state15 at grid coords:", state_to_pos(15))

  print("test pos_to_state")
  print("grid coords(0,0) is state:",pos_to_state((0,0)))
  print("grid coords(1,2) is state:",pos_to_state((1,2)))
  print("grid coords(2,2) is state:",pos_to_state((2,2)))
  print("grid coords(3,3) is state:",pos_to_state((3,3)))

  current_pos=(0,3)
  # print("current_pos:", current_pos, "up:",goUp(current_pos))
  # print("current_pos:", current_pos, "down:",goDown(current_pos))
  # print("current_pos:", current_pos, "left", goLeft(current_pos))
  # print("current_pos:", current_pos, "right", goRight(current_pos))


#test()

def goLeft(coord:tuple)->tuple:
  if (coord[1] - 1) < 0:
      return (coord[0], 0)
  return (coord[0], coord[1] -1)

def goRight(coord:tuple)->tuple:
  if (coord[1]+1) == grid_size:
    return (coord[0],grid_size-1)
  return(coord[0], coord[1]+1)

def goUp(coord:tuple)->tuple:
  if(coord[0]-1 < 0):
    return(0,coord[1])
  return (coord[0]-1, coord[1])

def goDown(coord:tuple)->tuple:
  if(coord[0]+1 == grid_size):
    return (grid_size-1,coord[1])
  return (coord[0]+1, coord[1])

# for a in actions:
#   print(a)


def getV_up(coord:tuple)->float:
  up = goUp(coord)
  return v[up[0]][up[1]]

def getV_down(coord:tuple)-> float:
  down = goDown(coord)
  return v[down[0]][down[1]]

def getV_left(coord:tuple)-> float:
  left = goLeft(coord)
  return v[left[0]][left[1]]

def getV_right(coord:tuple)-> float:
  right = goRight(coord)
  return v[right[0]][right[1]]


def is_terminal(current_pos:tuple)->bool:
  terminal_states=[(0,0),(3,3)]
  if current_pos in terminal_states:
    print("in terminal, settign reward to 0")
    return True
  return False

r = -1 
gamma = 1

def test_edges():
  # test the edge cases
  #current_pos=(0,0)
  current_pos=(3,0)
  #current_pos = (0,3)
  print("getV_up:",getV_up(current_pos),"current_pos:", current_pos, "up one:",goUp(current_pos))
  print("getV_down:",getV_down(current_pos),"current_pos:", current_pos, "down one:", goDown(current_pos))
  print("getV_left:", getV_left(current_pos), "current_pos:", current_pos, "left:", goLeft(current_pos))
  print("getV_right:", getV_right(current_pos),"current_pos:", current_pos, "goRight:", goRight(current_pos))

test_edges()

for idx in range(3):
  for i in range(grid_size):
    for j in range(grid_size):
      current_pos=(i,j)
      #print("state:",(i,j), "state: ",pos_to_state((i,j)))
      if is_terminal((i,j)):
        v_prime[current_pos[0]][current_pos[1]] = 0.
      else:
        #print(getV_up(current_pos), getV_down(current_pos), getV_left(current_pos), getV_right(current_pos))
        v_prime[current_pos[0]][current_pos[1]] = \
          0.25*(r+gamma*getV_up(current_pos)) \
          +0.25*(r+gamma*getV_down(current_pos)) \
          +0.25*(r+gamma*getV_left(current_pos)) \
          +0.25*(r+gamma*getV_right(current_pos))

      #print("-"*20)
  v = v_prime
  print(f"idx:{idx} V':{v_prime}")


#v_prime_1_0 = .25*(up)+.25*(down)+.25*(left)+.25*(right)