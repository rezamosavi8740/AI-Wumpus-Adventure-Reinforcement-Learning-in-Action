import argparse
import agent
# import environment
import runner
import Wumpus_Env
import pygame
import sys
import time

parser = argparse.ArgumentParser(description='RL running machine')
parser.add_argument('--environment', metavar='ENV_CLASS', type=str, default='WumpusWorldEnv', help='Class to use for the environment. Must be in the \'environment\' module')
parser.add_argument('--agent', metavar='AGENT_CLASS', default='Agent', type=str,
                    help='Class to use for the agent. Must be in the \'agent\' module.')
parser.add_argument('--ngames', type=int, metavar='n', default='30000', help='number of games to simulate')
parser.add_argument('--niter', type=int, metavar='n', default='25', help='max number of iterations per game')
parser.add_argument('--batch', type=int, metavar='nagent', default=None,
                    help='batch run several agent at the same time')
parser.add_argument('--verbose', action='store_true', help='Display cumulative results at each step')


# def step(environment, agent):
#     observation = environment._get_observation()
#     action = agent.act(observation)
#     (ob, reward, stop) = environment.step(action)
#     agent.reward(observation, action, reward)
#     return (observation, action, reward, stop)


def main():
    args = parser.parse_args()
    print('environment.{}'.format(args.environment))
    agent_class = eval('agent.{}'.format(args.agent))
    env_class = eval('Wumpus_Env.{}'.format(args.environment))

    # env_class = WumpusWorldEnv()
    # env_class.reset()

    print("Running a single instance simulation...")
    args.verbose = True
    my_runner = runner.Runner(env_class(), agent_class(), args.verbose)
    final_reward, env, ag = my_runner.loop(args.ngames, args.niter)
    print("Obtained a final reward of {}".format(final_reward))

    env.reset()

    #print(observation)
    while True:
        env.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        observation = env.get_observation()
        print(observation)
        action = ag.act(observation)
        time.sleep(1)
        (ob, reward, stop, done) = env.step(action)
        #observation = ob
        print(reward)
        print("action : " + str(action))
        print(ag.q_table)
        if stop:
            print("End game")
            time.sleep(1)
            break


if __name__ == "__main__":
    main()

# import main2
