import numpy as np

import torch

from agents.DQN import Agent as DQN_Agent
from networks.network_bodies import SimpleBody, AtariBody
from networks.networks import CategoricalDQN, CategoricalDuelingDQN


class Agent(DQN_Agent):
    def __init__(self, env=None, config=None, log_dir='/tmp/gym', tb_writer=None):
        super(Agent, self).__init__(env=env, config=config, log_dir=log_dir, tb_writer=tb_writer)

        self.supports = torch.linspace(self.config.c51_vmin, self.config.c51_vmax, self.config.c51_atoms).view(1, 1, self.config.c51_atoms).to(config.device)
        self.delta = (self.config.c51_vmax - self.config.c51_vmin) / (self.config.c51_atoms - 1)

    def declare_networks(self):
        if self.config.dueling_dqn:
            self.model = CategoricalDuelingDQN(self.num_feats, self.num_actions, noisy=self.config.noisy_nets, sigma_init=self.config.sigma_init, body=AtariBody, atoms=self.config.c51_atoms)
            self.target_model = CategoricalDuelingDQN(self.num_feats, self.num_actions, noisy=self.config.noisy_nets, sigma_init=self.config.sigma_init, body=AtariBody, atoms=self.config.c51_atoms)
        else:
            self.model = CategoricalDQN(self.num_feats, self.num_actions, noisy=self.config.noisy_nets, sigma_init=self.config.sigma_init, body=AtariBody, atoms=self.config.c51_atoms)
            self.target_model = CategoricalDQN(self.num_feats, self.num_actions, noisy=self.config.noisy_nets, sigma_init=self.config.sigma_init, body=AtariBody, atoms=self.config.c51_atoms)

    def projection_distribution(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        with torch.no_grad():
            max_next_dist = torch.zeros((self.config.batch_size, 1, self.config.c51_atoms), device=self.config.device, dtype=torch.float) + 1. / self.config.c51_atoms
            if not empty_next_state_values:
                self.target_model.sample_noise()
                if self.config.double_dqn:
                    max_next_actions = torch.argmax((self.model(non_final_next_states)*self.supports).sum(dim=2), dim=1)
                    max_next_dist[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_next_actions)
                else:
                    next_dist = self.target_model(non_final_next_states) * self.supports
                    max_next_dist[non_final_mask] = next_dist.sum(dim=2).max(dim=1)[0].view(non_final_next_states.size(0), 1, 1).expand(-1, -1, self.config.c51_atoms)
                max_next_dist = max_next_dist.squeeze()

            Tz = batch_reward.view(-1, 1) + (self.config.gamma**self.config.N_steps) * self.supports.view(1, -1) * non_final_mask.to(torch.float).view(-1, 1)
            Tz = Tz.clamp(self.config.c51_vmin, self.config.c51_vmax)
            b = (Tz - self.config.c51_vmin) / self.delta
            l = b.floor().to(torch.int64)
            u = b.ceil().to(torch.int64)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.config.c51_atoms - 1)) * (l == u)] += 1
            
            offset = torch.linspace(0, (self.config.batch_size - 1) * self.config.c51_atoms, self.config.batch_size).unsqueeze(dim=1).expand(self.config.batch_size, self.config.c51_atoms).to(batch_action)
            m = batch_state.new_zeros(self.config.batch_size, self.config.c51_atoms)
            m.view(-1).index_add_(0, (l + offset).view(-1), (max_next_dist * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1), (max_next_dist * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        return m
    
    def compute_loss(self, batch_vars, tstep):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        batch_action = batch_action.unsqueeze(dim=-1).expand(-1, -1, self.config.c51_atoms)
        batch_reward = batch_reward.view(-1, 1, 1)

        # estimate
        self.model.sample_noise()
        current_dist = self.model(batch_state).gather(1, batch_action).squeeze()

        target_prob = self.projection_distribution(batch_vars)
          
        loss = -(target_prob * current_dist.log()).sum(-1)
        if self.config.priority_replay:
            self.memory.update_priorities(indices, loss.detach().squeeze().abs().cpu().numpy().tolist())
            loss = loss * weights
        loss = loss.mean()

        return loss

    def get_action(self, s, eps):
        with torch.no_grad():
            if self.first_action:
                self.add_graph(s)

            if np.random.random() > eps or self.config.noisy_nets:
                X = torch.from_numpy(s).to(self.config.device).to(torch.float).view((-1,)+self.num_feats)
                X = X if self.config.s_norm is None else X/self.config.s_norm

                self.model.sample_noise()
                a = (self.model(X) * self.supports).sum(dim=2)
                return torch.argmax(a, dim=1).cpu().numpy()
            else:
                return np.random.randint(0, self.num_actions, (s.shape[0]))