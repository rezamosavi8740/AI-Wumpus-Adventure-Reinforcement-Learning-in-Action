import gym
from gym import spaces
import numpy as np
import pygame
import sys
import time


class WumpusWorldEnv(gym.Env):
    def __init__(self, size=4, number_of_pits=3):
        super(WumpusWorldEnv, self).__init__()

        self.size = size
        self.N_spares = 1
        self.screen_size = 400
        self.number_of_pits = number_of_pits
        self.cell_size = self.screen_size // self.size
        self.grid = np.zeros((size, size))

        rand_idx = np.random.choice(self.size*self.size, self.number_of_pits+3, replace=False)
        rand_pos = [(n%self.size, n//self.size) for n in rand_idx]
        self.pit_pos = rand_pos[0:self.number_of_pits]
        self.wumpus_pos = rand_pos[-3]
        self.init_wumpus_pos = self.wumpus_pos
        self.gold_pos = rand_pos[-2]
        self.agent_pos = rand_pos[-1]
        self.init_agent_pos = self.agent_pos

        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Tuple((spaces.Discrete(n=size, start=0), spaces.Discrete(n=size, start=0),
                                               spaces.Discrete(2), spaces.Discrete(2),
                                               spaces.Discrete(self.N_spares + 1, start=0)))

        self.done = False

        self.images = {
            'agent': pygame.transform.scale(pygame.image.load('./images/agent.png'), (self.cell_size, self.cell_size)),
            'wumpus': pygame.transform.scale(pygame.image.load('./images/wumpus.png'),
                                             (self.cell_size, self.cell_size)),
            'gold': pygame.transform.scale(pygame.image.load('./images/gold.png'), (self.cell_size, self.cell_size)),
            'pit': pygame.transform.scale(pygame.image.load('./images/pit.png'), (self.cell_size, self.cell_size)),
        }

        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Wumpus World")

    def _is_valid_move(self, pos):
        x, y = pos
        return 0 <= x < self.size and 0 <= y < self.size

    def _update_agent_position(self, action):
        """
            0 -> Up
            1 -> Down
            2 -> Left
            3 -> Right
            4 -> Shoot Up
            5 -> Shoot Down
            6 -> Shoot Left
            7 -> Shoot Right
        """
        x, y = self.agent_pos
        if action == 0:
            if self._is_valid_move((x - 1, y)):
                self.agent_pos = (x - 1, y)
                return 0
            else:
                return -2
        elif action == 1:
            if self._is_valid_move((x + 1, y)):
                self.agent_pos = (x + 1, y)
                return 0
            else:
                return -2
        elif action == 2:
            if self._is_valid_move((x, y - 1)):
                self.agent_pos = (x, y - 1)
                return 0
            else:
                return -2
        elif action == 3:
            if self._is_valid_move((x, y + 1)):
                self.agent_pos = (x, y + 1)
                return 0
            else:
                return -2
        elif action == 4:
            if self.N_spares > 0:
                self.N_spares -= 1
                if self.agent_pos[1] == self.wumpus_pos[1] and self.wumpus_pos[0] > self.agent_pos[0]:
                    self.wumpus_pos = (-10, -10)
                    return 1
                else:
                    return -1
        elif action == 5:
            if self.N_spares > 0:
                self.N_spares -= 1
                if self.agent_pos[1] == self.wumpus_pos[1] and self.wumpus_pos[0] < self.agent_pos[0]:
                    self.wumpus_pos = (-10, -10)
                    return 1
                else:
                    return -1
        elif action == 6:
            if self.N_spares > 0:
                self.N_spares -= 1
                if self.agent_pos[0] == self.wumpus_pos[0] and self.wumpus_pos[1] < self.agent_pos[1]:
                    self.wumpus_pos = (-10, -10)
                    return 1
                else:
                    return -1
        elif action == 7:
            if self.N_spares > 0:
                self.N_spares -= 1
                if self.agent_pos[0] == self.wumpus_pos[0] and self.wumpus_pos[1] > self.agent_pos[1]:
                    self.wumpus_pos = (-10, -10)
                    return 1
                else:
                    return -1

    def step(self, action):
        if self.done:
            print("Episode has ended. Call reset() to start a new episode.")
            self.reset()

        output = self._update_agent_position(action)
        state = self.get_observation()
        reward, stop = self._calculate_reward(output ,state)

        return state, reward, stop, self.done

    def _calculate_reward(self, output ,state):
        x, y = self.agent_pos

        if self.agent_pos == self.wumpus_pos:
            self.done = True
            return -100, 'Lose wumpus_po'  # Penalty for falling into pit or encountering Wumpus
        elif self.agent_pos in self.pit_pos:
            self.done = True
            return -100, 'Lose pit_pos'
        elif output == 1:
            return 50, None
        elif output == -1:
            return -50, None
        elif output == -2:
            return  -50, None
        elif self.agent_pos == self.gold_pos:
            self.done = True
            if state[2]:
                return 10000, 'Win'  # Reward for grabbing gold
            else:
                return 1000, 'Win'  # Reward for grabbing gold

        else:
            return -20, None  # Small penalty for each step

    def get_observation(self):  # (pos_x, pos_y, breeze, stinks)
        pos_x = self.agent_pos[0]
        pos_y = self.agent_pos[1]
        is_stink = (np.abs(self.agent_pos[0] - self.wumpus_pos[0]) + np.abs(self.agent_pos[1] - self.wumpus_pos[1]) <= 1)
        is_breeze = False
        for pit_pos in self.pit_pos:
            if np.abs(self.agent_pos[0] - pit_pos[0]) + np.abs(self.agent_pos[1] - pit_pos[1]) <= 1:
                is_breeze = True
                break

        return (pos_x, pos_y), is_stink, is_breeze, self.N_spares

    def reset(self):
        self.agent_pos = self.init_agent_pos
        self.done = False
        self.N_spares = 1
        self.wumpus_pos = self.init_wumpus_pos
        return self.get_observation()

    def _draw_grid(self):
        for i in range(1, self.size):
            pygame.draw.line(self.screen, (0, 0, 0), (i * self.cell_size, 0), (i * self.cell_size, self.screen_size))
            pygame.draw.line(self.screen, (0, 0, 0), (0, i * self.cell_size), (self.screen_size, i * self.cell_size))

    def render(self, mode='human', close=False):
        # Implement rendering if needed
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        self.screen.fill((255, 255, 255))
        for i in range(self.size):
            for j in range(self.size):
                x = j * self.cell_size
                y = i * self.cell_size

                if (i, j) == self.agent_pos:
                    self.screen.blit(self.images['agent'], (x, y))
                elif (i, j) == self.wumpus_pos:
                    self.screen.blit(self.images['wumpus'], (x, y))
                elif (i, j) == self.gold_pos:
                    self.screen.blit(self.images['gold'], (x, y))
                elif (i, j) in self.pit_pos:
                    self.screen.blit(self.images['pit'], (x, y))
                else:
                    pygame.draw.rect(self.screen, (255, 255, 255), (x, y, self.cell_size, self.cell_size))
        self._draw_grid()
        pygame.display.flip()