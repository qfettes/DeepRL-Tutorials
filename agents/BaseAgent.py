import numpy as np
import pickle
import os.path
import csv

import torch
import torch.optim as optim


class BaseAgent(object):
    def __init__(self, config, env, log_dir='/tmp/gym'):
        self.model=None
        self.target_model=None
        self.optimizer = None
        
        self.td_file = open(os.path.join(log_dir, 'td.csv'), 'a')
        self.td = csv.writer(self.td_file)

        self.sigma_parameter_mag_file = open(os.path.join(log_dir, 'sig_param_mag.csv'), 'a')
        self.sigma_parameter_mag = csv.writer(self.sigma_parameter_mag_file)

        self.rewards = []

        self.action_log_frequency = config.ACTION_SELECTION_COUNT_FREQUENCY
        self.action_selections = [0 for _ in range(env.action_space.n)]
        self.action_log_file = open(os.path.join(log_dir, 'action_log.csv'), 'a')
        self.action_log = csv.writer(self.action_log_file)

    def __del__(self):
        self.td_file.close()
        self.sigma_parameter_mag_file.close()
        self.action_log_file.close()

    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)

    def MSE(self, x):
        return 0.5 * x.pow(2)

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

    def save_sigma_param_magnitudes(self, tstep):
        with torch.no_grad():
            sum_, count = 0.0, 0.0
            for name, param in self.model.named_parameters():
                if param.requires_grad and 'sigma' in name:
                    sum_+= torch.sum(param.abs()).item()
                    count += np.prod(param.shape)
            
            if count > 0:
                self.sigma_parameter_mag.writerow((tstep, sum_/count))

    def save_td(self, td, tstep):
        self.td.writerow((tstep, td))

    def save_reward(self, reward):
        self.rewards.append(reward)

    def save_action(self, action, tstep):
        self.action_selections[int(action)] += 1.0/self.action_log_frequency
        if (tstep+1) % self.action_log_frequency == 0:
            self.action_log.writerow(list([tstep]+self.action_selections))
            self.action_selections = [0 for _ in range(len(self.action_selections))]

    def flush_data(self):
        self.action_log_file.flush()
        self.sigma_parameter_mag_file.flush()
        self.td_file.flush()
