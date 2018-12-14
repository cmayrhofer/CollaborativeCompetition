# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg_agent import DDPGAgent
import torch
import random
import torch.nn as nn
from utilities import soft_update, transpose_to_tensor, transpose_list
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'



class MultiAgentDDPG:
    def __init__(self, num_agents, state_size, action_size, random_seed=None):
        super(MultiAgentDDPG, self).__init__()
        """Initialize an Agent object.
        
        Params
        ======
            num_agents (int): number of agents 
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        if random_seed == None:
            random_seed = random.randint(1, 1000)
        self.seed = random.seed(random_seed)
        
        self.maddpg_agent = [DDPGAgent(state_size=self.state_size, action_size=self.action_size, random_seed=self.seed)\
                                for i in range(self.num_agents)]
        
        self.iter = 0

    def act(self, states):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs) for agent, obs in zip(self.maddpg_agent, states)]
        return actions
        
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        for agent, state, action, reward, next_state, done in zip(self.maddpg_agent, states, actions, rewards, next_states, dones):
            agent.step(state, action, reward, next_state, done)

            
            
            




