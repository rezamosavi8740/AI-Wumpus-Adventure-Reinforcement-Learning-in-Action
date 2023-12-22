from __future__ import division
from __future__ import print_function
import numpy as np


"""
This file contains the definition of the environment
in which the agents are run.
"""

ACT_UP    = 1
ACT_DOWN  = 2
ACT_LEFT  = 3
ACT_RIGHT = 4
ACT_TORCH_UP    = 5
ACT_TORCH_DOWN  = 6
ACT_TORCH_LEFT  = 7
ACT_TORCH_RIGHT = 8

ST_NOTHING = 0
ST_HOLE = 1
ST_HOLE

class Environment:
    # List of the possible actions by the agents
    possible_actions = []

    def __init__(self, gridsize=(6,6), num_wumpus=4, num_holes=3, charges=3):
        """Instanciate a new environement in its initial state.
        """
        self.gridsize = gridsize
        self.num_cases = gridsize[0]*gridsize[1]
        self.num_wumpus = num_wumpus
        self.num_holes = num_holes
        self.max_charges = charges
        # intial state
        rand_idx = np.random.choice(self.num_cases, self.num_wumpus+self.num_holes+2, replace=False)
        rand_pos = [(n%self.gridsize[1], n//self.gridsize[1]) for n in rand_idx]
        self.init_holes = rand_pos[0:self.num_holes]
        self.init_wumpus = rand_pos[self.num_holes:-2]
        self.init_treasure = rand_pos[-2]
        self.init_agent = rand_pos[-1]
        self.reset()

    def reset(self):
        """Reset the environment for a new run."""
        # place holes
        self.rem_charges = self.max_charges
        self.wumpus = list(self.init_wumpus)
        self.init_wampus_pos = list(self.init_wumpus)
        self.holes = list(self.init_holes)
        self.treasure = self.init_treasure
        self.agent = self.init_agent

    def generateMaze(self):

        rand_idx = np.random.choice(self.num_cases, self.num_wumpus + self.num_holes + 2, replace=False)
        rand_pos = [(n % self.gridsize[1], n // self.gridsize[1]) for n in rand_idx]
        self.init_holes = rand_pos[0:self.num_holes]
        self.init_wumpus = rand_pos[self.num_holes:-2]
        self.init_treasure = rand_pos[-2]
        self.init_agent = rand_pos[-1]


    def observe(self):
        """Returns the current observation that the agent can make
        of the environment, if applicable.
        """
        return (self.agent, self.is_near_wumpus(), self.is_near_hole(), self.rem_charges)

    def is_near_wumpus(self):
        (ax, ay) = self.agent
        for (nx, ny) in self.wumpus:
            if (ax-nx)**2 + (ay-ny)**2 <= 1:
                return True
        return False

    def is_near_hole(self):
        (ax, ay) = self.agent
        for (nx, ny) in self.holes:
            if (ax-nx)**2 + (ay-ny)**2 <= 1:
                return True
        return False

    def move_object(self, pos, action):
        (x, y) = pos
        if action == ACT_UP:
            y += 1
        elif action == ACT_DOWN:
            y -= 1
        elif action == ACT_LEFT:
            x -= 1
        elif action == ACT_RIGHT:
            x += 1
        flag = False
        if (x == -1 or x == self.gridsize[0]) or (y == -1 or y == self.gridsize[1]):
            flag = True
        x = max(0, min(self.gridsize[0]-1, x))
        y = max(0, min(self.gridsize[1]-1, y))
        if (x, y) == self.agent:
            (x, y) = pos
        return (x, y), flag

    def move_wampus(self, init_pos, pos, action):
        (a, b) = init_pos
        (x, y) = pos
        if action == ACT_UP:
            y += 1
        elif action == ACT_DOWN:
            y -= 1
        elif action == ACT_LEFT:
            x -= 1
        elif action == ACT_RIGHT:
            x += 1

        x = max(0, min(self.gridsize[0] - 1, x))
        y = max(0, min(self.gridsize[1] - 1, y))
        if (x, y) == self.agent or abs(a - x) == 2 or abs(b - y) == 2:
            (x, y) = pos
        return (x, y)

    def move_wumpus(self):
        self.wumpus = [self.move_wampus(init_pos, pos, act) for (init_pos, pos, act) in zip(self.init_wampus_pos, self.wumpus, np.random.randint(1, 5, self.num_wumpus)) ]

    def kill_wumpus_at(self, pos):
        if pos in self.wumpus:
            self.wumpus.remove(pos)
            return True
        return False

    def torchlight(self, action):
        if self.rem_charges <= 0:
            return False
        (x,y) = self.agent
        if action == ACT_TORCH_UP:
            self.rem_charges -= 1
            return self.kill_wumpus_at((x, y+1))
        elif action == ACT_TORCH_DOWN:
            self.rem_charges -= 1
            return self.kill_wumpus_at((x, y-1))
        elif action == ACT_TORCH_LEFT:
            self.rem_charges -= 1
            return self.kill_wumpus_at((x-1, y))
        elif action == ACT_TORCH_RIGHT:
            self.rem_charges -= 1
            return self.kill_wumpus_at((x+1, y))
        else:
            return False

    def act(self, action):
        """Perform given action by the agent on the environment,
        and returns a reward.
        """
        self.agent, flag = self.move_object(self.agent, action)
        if flag:
            return (-2.0, None)
        if self.agent in self.holes:
            return (-2.0, "Fell into a hole.")
        if self.agent in self.wumpus:
            return (-2.0, "Encountered a Wumpus.")
        if self.torchlight(action):
            return (0.250, None)
        if not self.torchlight(action) and 5 <= action <= 8:
            return (-0.250, None)
        if self.agent == self.treasure:
            return (100.0, "Found the treasure.")

        #self.move_wumpus()
        return (-0.10, None)

    def display(self):
        print("+-", end='')
        for x in range(self.gridsize[0]):
            print("--", end='')
        print("+ ")
        for y in range(self.gridsize[1]-1, -1, -1):
            print("| ", end='')
            for x in range(self.gridsize[0]):
                if (x,y) in self.wumpus:
                    print("W ", end='')
                elif (x, y) in self.holes:
                    print("O ", end='')
                elif (x, y) == self.treasure:
                    print("$ ", end='')
                elif (x, y) == self.agent:
                    print("A ", end='')
                else:
                    print(". ", end='')
            print("|")
        print("+-", end='')
        for x in range(self.gridsize[0]):
            print("--", end='')
        print("+")