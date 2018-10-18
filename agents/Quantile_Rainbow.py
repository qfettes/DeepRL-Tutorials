import numpy as np

import torch

from agents.DQN import Model as DQN_Agent
from networks.network_bodies import SimpleBody, AtariBody
from networks.networks import DuelingQRDQN
from utils.ReplayMemory import PrioritizedReplayMemory

class Model(DQN_Agent):
    def __init__(self, static_policy=False, env=None, config=None):
        self.num_quantiles = config.QUANTILES
        self.cumulative_density = torch.tensor((2 * np.arange(self.num_quantiles) + 1) / (2.0 * self.num_quantiles), device=config.device, dtype=torch.float) 
        self.quantile_weight = 1.0 / self.num_quantiles

        super(Model, self).__init__(static_policy, env, config)

        self.nsteps=max(self.nsteps, 3)
    
    
    def declare_networks(self):
        self.model = DuelingQRDQN(self.env.observation_space.shape, self.env.action_space.n, noisy=True, sigma_init=self.sigma_init, quantiles=self.num_quantiles)
        self.target_model = DuelingQRDQN(self.env.observation_space.shape, self.env.action_space.n, noisy=True, sigma_init=self.sigma_init, quantiles=self.num_quantiles)

    def declare_memory(self):
        self.memory = PrioritizedReplayMemory(self.experience_replay_size, self.priority_alpha, self.priority_beta_start, self.priority_beta_frames)

    def next_distribution(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        with torch.no_grad():
            quantiles_next = torch.zeros((self.batch_size, self.num_quantiles), device=self.device, dtype=torch.float)
            if not empty_next_state_values:
                self.target_model.sample_noise()
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                quantiles_next[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_next_action).squeeze(dim=1)

            quantiles_next = batch_reward + ((self.gamma**self.nsteps)*quantiles_next)

        return quantiles_next
    
    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        batch_action = batch_action.unsqueeze(dim=-1).expand(-1, -1, self.num_quantiles)

        self.model.sample_noise()
        quantiles = self.model(batch_state)
        quantiles = quantiles.gather(1, batch_action).squeeze(1)

        quantiles_next = self.next_distribution(batch_vars)
          
        diff = quantiles_next.t().unsqueeze(-1) - quantiles.unsqueeze(0)

        loss = self.huber(diff) * torch.abs(self.cumulative_density.view(1, -1) - (diff < 0).to(torch.float))
        loss = loss.transpose(0,1)
        self.memory.update_priorities(indices, loss.detach().mean(1).sum(-1).abs().cpu().numpy().tolist())
        loss = loss * weights.view(self.batch_size, 1, 1)
        loss = loss.mean(1).sum(-1).mean()

        return loss

    def get_action(self, s, eps):
        with torch.no_grad():
            X = torch.tensor([s], device=self.device, dtype=torch.float) 
            self.model.sample_noise()
            a = (self.model(X) * self.quantile_weight).sum(dim=2).max(dim=1)[1]
            return a.item()

    def get_max_next_state_action(self, next_states):
        next_dist = self.model(next_states) * self.quantile_weight
        return next_dist.sum(dim=2).max(1)[1].view(next_states.size(0), 1, 1).expand(-1, -1, self.num_quantiles)