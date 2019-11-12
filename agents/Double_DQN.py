import numpy as np

import torch

from agents.DQN import Agent as DQN_Agent

class Agent(DQN_Agent):
    def __init__(self, static_policy=False, env=None, config=None, log_dir='/tmp/gym'):
        super(Agent, self).__init__(static_policy, env, config, log_dir=log_dir)

    def get_max_next_state_action(self, next_states):
        return self.model(next_states).max(dim=1)[1].view(-1, 1) 
