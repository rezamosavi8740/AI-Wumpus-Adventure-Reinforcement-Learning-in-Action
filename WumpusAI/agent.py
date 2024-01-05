import numpy as np
import collections

"""
Contains the definition of the agent that will run in an
environment.
"""

ACT_UP = 1
ACT_DOWN = 2
ACT_LEFT = 3
ACT_RIGHT = 4
ACT_TORCH_UP = 5
ACT_TORCH_DOWN = 6
ACT_TORCH_LEFT = 7
ACT_TORCH_RIGHT = 8


class RandomAgent:
    def __init__(self):
        """
            learning rate value
            gamma value
        """
        self.learning_rate = .01  # learning rate
        self.gamma = .8

        self.epsilon_a = -1 / 1000
        self.epsilon_b = 1

        # State-action function (Q-function)
        self.q_table = {}

        # Episode is defined as each time we starting moving until we die
        self.n_episode = 0

        # Step is defined as the iteration in each game
        self.step = 0

        self.cumul_reward = 0

    def setAgent(self, q_table):
        self.q_table = q_table

    def saveAgent(self):
        return {
            "q_table": self.q_table
        }

    def reset(self):
        self.step = 0
        self.n_episode += 1
        if self.n_episode == 15000:
            print(self.cumul_reward)
            # print(self.q_table)

    def act(self, observation):
        self.step += 1
        self.epsilon = 0.5

        position, is_stink, is_breeze, N_spares = observation

        self.create_state(observation)
        state_action = self.q_table[observation]

        # if self.n_episode >= 3000:
        #  if smell and charges > 0:
        #      return self.choose_action(state_action)
        #   else:
        #       return self.choose_action(state_action[0:4])

        if np.random.uniform() >= (self.epsilon_a * self.epsilon + 1) ** self.n_episode:
            if is_stink and N_spares > 0:
                return self.choose_action(state_action)
            else:
                return self.choose_action(state_action[0:4])
        else:
            if is_stink and N_spares > 0:
                return np.random.randint(0, 8)
            else:
                return np.random.randint(0, 4)

    def reward(self, observation, action, reward):
        if self.step > 1:
            self.learn(s=observation,
                       a=action,
                       r=reward,
                       s_=self.next_observation(observation, action),
                       )
        if self.n_episode >= 3000:
            self.cumul_reward += reward

    def create_state(self, state):
        """
            If new state, create it in the q_table, with default values = 0
        """
        if state not in self.q_table.keys():
            self.q_table[state] = 8 * [0.5]

    def learn(self, s, a, r, s_):
        self.create_state(s_)  # create only if not exists
        q_predict = self.q_table[s][a]
        q_target = r + self.gamma * max(self.q_table[s_])
        self.q_table[s][a] += self.learning_rate * (q_target - q_predict)

    def choose_action(self, state_action):
        # get the list of actions with maximum value in Q table
        action_with_max_value = [i for i in range(len(state_action)) if state_action[i] == max(state_action)]
        return np.random.choice(action_with_max_value)

    def next_observation(self, observation, action):
        position, is_stink, is_breeze, N_spares = observation
        if action >= 4 and N_spares > 0:
            N_spares -= 1
            next_observation = (self.next_position(position, action), is_stink, is_breeze, N_spares)
        else:
            next_observation = (self.next_position(position, action), is_stink, is_breeze, N_spares)

        return next_observation

    def next_position(self, position, action):
        x, y = position
        if action == 0 and x >= 1:
            x -= 1
        elif action == 1 and x <= 2:
            x += 1
        elif action == 2 and x >= 1:
            y -= 1
        elif action == 3 and x <= 2:
            x += 1
        return x, y


Agent = RandomAgent
