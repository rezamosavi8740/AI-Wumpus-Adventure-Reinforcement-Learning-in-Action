
import argparse
import agent
import environment
import runner
from Wumpus_Env import WumpusWorldEnv
import pygame
import sys
import time

parser = argparse.ArgumentParser(description='RL running machine')
#parser.add_argument('--WumpusWorldEnv', metavar='ENV_CLASS', type=str, default='WumpusWorldEnv', help='Class to use for the environment. Must be in the \'environment\' module')
parser.add_argument('--agent', metavar='AGENT_CLASS', default='Agent', type=str, help='Class to use for the agent. Must be in the \'agent\' module.')
parser.add_argument('--ngames', type=int, metavar='n', default='100000', help='number of games to simulate')
parser.add_argument('--niter', type=int, metavar='n', default='100', help='max number of iterations per game')
parser.add_argument('--batch', type=int, metavar='nagent', default=None, help='batch run several agent at the same time')
parser.add_argument('--verbose', action='store_true', help='Display cumulative results at each step')


def step(environment ,agent):
    observation = environment._get_observation()
    action = agent.act(observation)
    (ob, reward, stop) = environment.step(action)
    agent.reward(observation, action, reward)
    return (observation, action, reward, stop)

def main():
    args = parser.parse_args()
    agent_class = eval('agent.{}'.format(args.agent))
    #env_class = eval('environment.{}'.format(args.environment))

    env_class = WumpusWorldEnv()

    if args.batch is not None:
        print("Running a batched simulation with {} agents in parallel...".format(args.batch))
        my_runner = runner.BatchRunner(env_class, agent_class, args.batch, args.verbose)
        final_reward ,env ,ag = my_runner.loop(args.ngames, args.niter)
        print("Obtained a final average reward of {}".format(final_reward))

    else:
        print("Running a single instance simulation...")
        args.verbose = True
        my_runner = runner.Runner(env_class, agent_class(), args.verbose)
        final_reward ,env ,ag= my_runner.loop(args.ngames, args.niter)
        print("Obtained a final reward of {}".format(final_reward))

    observation = env._get_observation()

    while True:

        env.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        action = ag.act(observation)
        time.sleep(0.5)
        (ob, reward, stop) = env.step(action)
        observation = ob
        print(reward)
        if stop :
            print("End game")
            time.sleep(1)
            break








if __name__ == "__main__":
    main()

#import main2
