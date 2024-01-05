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

                if stop is not None:
                    break
                # PYGame
            if self.verbose:
                print(" <=> Finished game number: {} <=>".format(g))
                print()


        return cumul_reward, self.environment, self.agent

    def findMap(self):
        Map = np.empty((self.environment.gridsize[0], self.environment.gridsize[1]))
        for x in range(self.environment.gridsize[0]):
            for y in range(self.environment.gridsize[1]):
                if (y, x) in self.environment.wumpus:
                    Map[(self.environment.gridsize[1] - 1) - x][y] = 2
                elif (y, x) in self.environment.holes:
                    Map[(self.environment.gridsize[1] - 1) - x][y] = 3
                elif (y, x) == self.environment.treasure:
                    Map[(self.environment.gridsize[1] - 1) - x][y] = 1
                elif (y, x) == self.environment.agent:
                    Map[(self.environment.gridsize[1] - 1) - x][y] = 4
                else:
                    Map[(self.environment.gridsize[1] - 1) - x][y] = 0

        return Map


def iter_or_loopcall(o, count):
    if callable(o):
        return [o() for _ in range(count)]
    else:
        # must be iterable
        return list(iter(o))


class BatchRunner:
    """
    Runs several instances of the same RL problem in parallel
    and aggregates the results.
    """

    def __init__(self, env_maker, agent_maker, count, verbose=False):
        self.environments = iter_or_loopcall(env_maker, count)
        self.agents = iter_or_loopcall(agent_maker, count)
        assert (len(self.agents) == len(self.environments))
        self.verbose = verbose
        self.ended = [False for _ in self.environments]

    def game(self, max_iter):
        rewards = []
        for (agent, env) in zip(self.agents, self.environments):
            agent.reset()
            env.reset()
            game_reward = 0
            for i in range(1, max_iter + 1):
                observation = env.observe()
                action = agent.act(observation)
                (reward, stop) = env.act(action)
                agent.reward(observation, action, reward)
                game_reward += reward
                if stop is not None:
                    break
            rewards.append(game_reward)
        return sum(rewards) / len(rewards)

    def loop(self, games, max_iter):
        cum_avg_reward = 0.0
        for g in range(1, games + 1):
            avg_reward = self.game(max_iter)
            if g > 9 * games / 10:
                cum_avg_reward += avg_reward
            if self.verbose:
                print("Simulation game {}:".format(g))
                print(" ->            average reward: {}".format(avg_reward))
                print(" -> cumulative average reward: {}".format(cum_avg_reward))
        return cum_avg_reward
