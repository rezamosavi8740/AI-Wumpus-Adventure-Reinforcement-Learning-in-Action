from agent import RandomAgent
from environment import Environment
import runner


class Display():
    def __int__(self ,SAgent ,SEnvironment):
        self.SAgent = SAgent
        self.SEnvironment = SEnvironment


    def setAll(self):
        agent = RandomAgent()
        agent.setAgent(self.SAgent["q_table"])
        self.agent = agent
        environment = Environment()
        environment.setEnvironment(
        self.SEnvironment['gridsize'],
        self.SEnvironment['num_wumpus'],
        self.SEnvironment['num_holes'],
        self.SEnvironment['charges'],
        self.SEnvironment['init_holes'],
        self.SEnvironment['init_wumpus'],
        self.SEnvironment['init_treasure'],
        self.SEnvironment['init_agent']
        )
        self.environment = environment


