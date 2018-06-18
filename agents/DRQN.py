import numpy as np

import torch

from agents.DQN import Model as DQN_Agent
from networks.networks import DRQN
from utils.ReplayMemory import RecurrentExperienceReplayMemory
from utils.hyperparameters import SEQUENCE_LENGTH, device

class Model(DQN_Agent):
    def __init__(self, static_policy=False, env=None):
        self.sequence_length=SEQUENCE_LENGTH

        super(Model, self).__init__(static_policy, env)

        self.reset_hx()
    
    
    def declare_networks(self):
        self.model = DRQN(self.env.observation_space.shape, self.env.action_space.n, noisy=self.noisy, sigma_init=self.sigma_init)
        self.target_model = DRQN(self.env.observation_space.shape, self.env.action_space.n, noisy=self.noisy, sigma_init=self.sigma_init)

    def declare_memory(self):
        self.memory = RecurrentExperienceReplayMemory(self.experience_replay_size, self.sequence_length)

    def prep_minibatch(self):
        transitions, indices, weights = self.memory.sample(self.batch_size)

        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        shape = (self.batch_size,self.sequence_length)+self.num_feats

        batch_state = torch.tensor(batch_state, device=device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=device, dtype=torch.long).view(self.batch_size, self.sequence_length, -1)[:,self.sequence_length-1,:]
        batch_reward = torch.tensor(batch_reward, device=device, dtype=torch.float).view(self.batch_size, self.sequence_length, -1)[:,self.sequence_length-1,:]
        #get set of next states for end of each sequence
        batch_next_state = tuple([batch_next_state[i] for i in range(len(batch_next_state)) if (i+1)%(self.sequence_length)==0])

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=device, dtype=torch.uint8)
        try: #sometimes all next states are false, especially with nstep returns
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=device, dtype=torch.float).unsqueeze(dim=1)
            empty_next_state_values = False
        except:
            empty_next_state_values = True

        #non_final_next_states = torch.cat([batch_state[non_final_mask, 1:, :], non_final_next_states], dim=1)

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights

    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        #estimate
        self.model.sample_noise()
        current_q_values, hx = self.model(batch_state)
        hx = hx[non_final_mask]
        current_q_values = current_q_values.gather(1, batch_action)
        
        #target
        with torch.no_grad():
            max_next_q_values = torch.zeros(self.batch_size, device=device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states, hx)
                self.target_model.sample_noise()
                max_next, _ = self.target_model(non_final_next_states, hx)
                max_next_q_values[non_final_mask] = max_next.gather(1, max_next_action)
            expected_q_values = batch_reward + ((self.gamma**self.nsteps)*max_next_q_values)

        diff = (expected_q_values - current_q_values)
        loss = self.huber(diff)
        loss = loss.mean()

        return loss

    def get_action(self, s, eps=0.1):
        with torch.no_grad():
            if np.random.random() >= eps or self.static_policy or self.noisy:
                X = torch.tensor([[s]], device=device, dtype=torch.float) 
                self.model.sample_noise()
                a, self.action_hx = self.model(X, self.action_hx)
                a = a.max(1)[1]
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)

    def get_max_next_state_action(self, next_states, hx):
        max_next, _ = self.target_model(next_states, hx)
        return max_next.max(dim=1)[1].view(-1, 1)

    def reset_hx(self):
        self.action_hx = self.model.init_hidden(1)

    