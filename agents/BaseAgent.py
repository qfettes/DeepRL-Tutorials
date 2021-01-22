import numpy as np
import pickle
import os.path
import csv

import torch
import torch.optim as optim


class BaseAgent(object):
    def __init__(self, env, config, log_dir='/tmp/gym', tb_writer=None):
        self.q_func=None
        self.target_q_func=None
        self.optimizer = None

        self.log_dir = log_dir
        self.tb_writer=tb_writer
        
        # # upload hparms to tensorboard
        # log_config = config.__dict__.copy()
        # tmp_device = log_config['device']
        # log_config['device'] = 0
        # tmp_seed = log_config['seed']
        # log_config['seed'] = -1 if log_config['seed'] is None else log_config['seed']
        # tensor_type = None
        # for key in log_config.keys():
        #     if type(log_config[key]) == list:
        #         log_config[key] = torch.tensor(log_config[key]).to(torch.float16)
        #         tensor_type = type(log_config[key])
        # self.tb_writer.add_hparams(log_config, {'Performance/Agent Reward':0.0, 'Performance/Environment Reward':0})

        self.rewards = []

        # self.action_selections = [0 for _ in range(env.action_space.n)]

    def save_w(self):
        torch.save(self.q_func.state_dict(), os.path.join(self.log_dir, 'saved_model', 'model.dump'))
        torch.save(self.optimizer.state_dict(), os.path.join(self.log_dir, 'saved_model', 'optim.dump'))
    
    def load_w(self):
        fname_model = os.path.join(self.log_dir, 'saved_model', 'model.dump')
        fname_optim = os.path.join(self.log_dir, 'saved_model', 'optim.dump')

        if os.path.isfile(fname_model):
            self.q_func.load_state_dict(torch.load(fname_model))

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
            for name, param in self.q_func.named_parameters():
                if param.requires_grad and 'sigma' in name:
                    sum_+= torch.sum(param.abs()).item()
                    count += np.prod(param.shape)
            
            if count > 0:
                with open(os.path.join(self.log_dir, 'sig_param_mag.csv'), 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow((tstep, sum_/count))

    def save_reward(self, reward):
        self.rewards.append(reward)
    
    def count_parameters(self, model):
        if model is None:
            return 0
        else:
            total = 0
            for name, param in model.state_dict().items():
                total += np.prod(param.shape)
            return total

    def save_generic_stat(self, stat, tstep, stat_name):
        with open(os.path.join(self.log_dir, 'logs', stat_name+'.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow((tstep, stat))
