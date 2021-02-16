import numpy as np
import torch
from networks.network_bodies import AtariBody, SimpleBody
from networks.networks import CategoricalDQN, CategoricalDuelingDQN

from agents.DQN import Agent as DQN_Agent


class Agent(DQN_Agent):
    def __init__(self, env=None, config=None, log_dir='/tmp/gym', tb_writer=None):
        super().__init__(env=env, config=config,
                                    log_dir=log_dir, tb_writer=tb_writer)

        self.supports = torch.linspace(self.config.c51_vmin, self.config.c51_vmax,
                                       self.config.c51_atoms, device=config.device).view(1, 1, self.config.c51_atoms)
        self.delta = (self.config.c51_vmax - self.config.c51_vmin) / \
            (self.config.c51_atoms - 1)

    def declare_networks(self):
        if self.config.dueling_dqn:
            self.q_net = CategoricalDuelingDQN(self.num_feats, self.num_actions, noisy=self.config.noisy_nets,
                                               sigma_init=self.config.sigma_init, body=AtariBody, atoms=self.config.c51_atoms)
            self.target_q_net = CategoricalDuelingDQN(self.num_feats, self.num_actions, noisy=self.config.noisy_nets,
                                                      sigma_init=self.config.sigma_init, body=AtariBody, atoms=self.config.c51_atoms)
        else:
            self.q_net = CategoricalDQN(self.num_feats, self.num_actions, noisy=self.config.noisy_nets,
                                        sigma_init=self.config.sigma_init, body=AtariBody, atoms=self.config.c51_atoms)
            self.target_q_net = CategoricalDQN(self.num_feats, self.num_actions, noisy=self.config.noisy_nets,
                                               sigma_init=self.config.sigma_init, body=AtariBody, atoms=self.config.c51_atoms)

    def projection_distribution(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        with torch.no_grad():
            self.target_q_net.sample_noise()
            max_next_dist = torch.zeros((self.config.batch_size, 1, self.config.c51_atoms),
                                        device=self.device, dtype=torch.float) + 1. / self.config.c51_atoms

            if not empty_next_state_values:
                if self.config.double_dqn:
                    max_next_actions = torch.argmax((self.q_net(non_final_next_states)[
                                                    0]*self.supports).sum(dim=2), dim=1)
                    max_next_dist[non_final_mask] = self.target_q_net(
                        non_final_next_states)[0].gather(1, max_next_actions)
                else:
                    next_probs = (self.target_q_net(non_final_next_states)[
                                  0] * self.supports).sum(-1)
                    max_next_dist[non_final_mask] = next_probs.max(
                        -1)[0].view(-1, 1, 1).expand(-1, -1, self.config.c51_atoms)
                max_next_dist = max_next_dist.squeeze()

            Tz = batch_reward.view(-1, 1) + (self.config.gamma**self.config.N_steps) * \
                self.supports.view(1, -1) * \
                non_final_mask.to(torch.float).view(-1, 1)
            Tz = Tz.clamp(self.config.c51_vmin, self.config.c51_vmax)
            b = (Tz - self.config.c51_vmin) / self.delta
            l = b.floor().to(torch.int64)
            u = b.ceil().to(torch.int64)

            # NOTE: New from modular drl
            d_m_l = (u + (l == u).float() - b) * max_next_dist
            d_m_u = (b - l) * max_next_dist
            target_prob = torch.zeros(max_next_dist.size(
            ), device=self.device, dtype=torch.float)
            for i in range(target_prob.size(0)):
                target_prob[i].index_add_(0, l[i], d_m_l[i])
                target_prob[i].index_add_(0, u[i], d_m_u[i])

            # l[(u > 0) * (l == u)] -= 1
            # u[(l < (self.config.c51_atoms - 1)) * (l == u)] += 1

            # offset = torch.linspace(0, (self.config.batch_size - 1) * self.config.c51_atoms, self.config.batch_size).unsqueeze(dim=1).expand(self.config.batch_size, self.config.c51_atoms).to(batch_action)
            # m = batch_state.new_zeros(self.config.batch_size, self.config.c51_atoms)
            # m.view(-1).index_add_(0, (l + offset).view(-1), (max_next_dist * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            # m.view(-1).index_add_(0, (u + offset).view(-1), (max_next_dist * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        return target_prob

    def compute_loss(self, batch_vars, tstep):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        batch_action = batch_action.unsqueeze(
            dim=-1).expand(-1, -1, self.config.c51_atoms)
        batch_reward = batch_reward.view(-1, 1, 1)

        # estimate
        self.q_net.sample_noise()
        log_prob = self.q_net(batch_state)[1].gather(1, batch_action).squeeze()

        target_prob = self.projection_distribution(batch_vars)

        loss = -(target_prob * log_prob).sum(-1)
        if self.config.priority_replay:
            self.memory.update_priorities(
                indices, loss.detach().squeeze().abs().cpu().numpy().tolist())
            loss = loss * weights
        loss = loss.mean()

        return loss

    def get_action(self, s, eps):
        with torch.no_grad():
            if self.first_action:
                self.add_graph(s)

            if np.random.random() > eps or self.config.noisy_nets:
                X = torch.from_numpy(s).to(self.device).to(
                    torch.float).view((-1,)+self.num_feats)

                self.q_net.sample_noise()
                a = (self.q_net(X)[0] * self.supports).sum(dim=2)
                return torch.argmax(a, dim=1).cpu().numpy()
            else:
                return np.random.randint(0, self.num_actions, (s.shape[0]))
