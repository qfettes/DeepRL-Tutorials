import numpy as np

import torch

from agents.DQN import Model as DQN_Agent
from networks.networks import DuelingDQN, DuelingDQN_simple

class Model(DQN_Agent):
    def __init__(self, static_policy=False, env=None):
        super(Model, self).__init__(static_policy, env)