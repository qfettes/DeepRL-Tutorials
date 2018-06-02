import numpy as np

import torch

from agents.DQN import Model as DQN_Agent
from networks.networks import DuelingDQN, DuelingDQN_simple

class Model(DQN_Agent):
    def __init__(self, static_policy=False, env=None):
        super(Model, self).__init__(static_policy, env)

    def declare_networks(self):
        self.model = DuelingDQN_simple(self.env.observation_space.shape, self.env.action_space.n)
        self.target_model = DuelingDQN_simple(self.env.observation_space.shape, self.env.action_space.n)