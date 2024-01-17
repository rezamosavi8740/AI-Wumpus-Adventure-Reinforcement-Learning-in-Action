"""
This is the machinnery that runs your agent in an environment.

This is not intented to be modified during the practical.
"""

import numpy as np


class Runner:
    def __init__(self, environment, agent, verbose=False):
        self.environment = environment
        self.agent = agent
        self.verbose = verbose

    def step(self):
        observation = self.environment.get_observation()
        action = self.agent.act(observation)
        (ob, reward, stop, done) = self.environment.step(action)
        self.agent.reward(observation, action, reward)
        return ob, action, reward, stop

    def loop(self, games, max_iter):
        cumul_reward = 0.0
        # SEnvironment = self.environment.saveEnvironment()
        # SAgent = self.agent.saveAgent()

        for g in range(1, games + 1):
            self.agent.reset()
            self.environment.reset()
            for i in range(1, max_iter + 1):
                print(i)
                (obs, act, rew, stop) = self.step()
                cumul_reward += rew


                if stop is not None:
                    print(" ->    Terminal event: {}".format(stop))
                    print()
                    break
                # PYGame
            if self.verbose:
                print(" <=> Finished game number: {} <=>".format(g))
                print()


        return cumul_reward, self.environment, self.agent