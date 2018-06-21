import numpy as np
import pickle
import os.path

import torch
import torch.optim as optim


class BaseAgent(object):
    def __init__(self):
        self.model=None
        self.target_model=None
        self.optimizer = None
        self.losses = []
        self.rewards = []
        self.sigma_parameter_mag=[]

    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)

    def save_w(self):
        torch.save(self.model.state_dict(), './saved_agents/model.dump')
        torch.save(self.optimizer.state_dict(), './saved_agents/optim.dump')
    
    def load_w(self):
        fname_model = "./saved_agents/model.dump"
        fname_optim = "./saved_agents/optim.dump"

        if os.path.isfile(fname_model):
            self.model.load_state_dict(torch.load(fname_model))
            self.target_model.load_state_dict(self.model.state_dict())

        if os.path.isfile(fname_optim):
            self.optimizer.load_state_dict(torch.load(fname_optim))

    def save_replay(self):
        pickle.dump(self.memory, open('./saved_agents/exp_replay_agent.dump', 'wb'))

    def load_replay(self):
        fname = './saved_agents/exp_replay_agent.dump'
        if os.path.isfile(fname):
            self.memory = pickle.load(open(fname, 'rb'))

    def save_sigma_param_magnitudes(self):
        tmp = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'sigma' in name:
                    tmp+=param.data.cpu().numpy().ravel().tolist()
        if tmp:
            self.sigma_parameter_mag.append(np.mean(np.abs(np.array(tmp))))

    def save_loss(self, loss):
        self.losses.append(loss)

    def save_reward(self, reward):
        self.rewards.append(reward)