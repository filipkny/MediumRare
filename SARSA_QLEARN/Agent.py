import copy
import numpy as np

class Agent(object):
    def __init__(self,init_y,init_x):
        self.pos = [init_y,init_x]
        self.reward = 0

    def select_e_greedily(self, Qmat, e = 0.2):
        [y,x] = self.pos
        choices = Qmat[y][x]
        if np.random.uniform(0,1) < e:
            # Select random move
            return np.random.choice(list(choices.keys()))
        else:
            # Select highest Q move
            return max(choices, key=choices.get)

    def move(self, dir, walls, size):
        new_pos = copy.copy(self.pos)

        if dir == 'north':
            new_pos[0] += -1
        elif dir == 'south':
            new_pos[0] += 1
        elif dir == 'east':
            new_pos[1] += 1
        elif dir == 'west':
            new_pos[1] += -1

        # Check if bumped against a wall or out-of bounds
        bumped = new_pos in walls or \
                 new_pos[0] < 0 or \
                 new_pos[0] >= size or \
                 new_pos[1] < 0 or \
                 new_pos[1] >= size

        if not bumped:
            self.pos = new_pos

        return self.pos

