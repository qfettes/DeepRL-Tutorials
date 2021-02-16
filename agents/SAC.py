from __future__ import absolute_import

import sys
import itertools
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
# from networks.networks import DQN_SAC, Actor_SAC
from networks.networks import MLPActorCritic as actor_critic
from utils.ReplayMemory import ExperienceReplayMemory, PrioritizedReplayMemory

from agents.BaseAgent import BaseAgent
from agents.DQN import Agent as DQN_Agent

import random


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size,) + obs_dim, dtype=np.float32)
        self.obs2_buf = np.zeros((size,) + obs_dim, dtype=np.float32)
        self.act_buf = np.zeros((size,) + (act_dim,), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

class Agent(DQN_Agent):
    def __init__(self, env=None, config=None, log_dir='/tmp/gym', tb_writer=None):
        # NOTE: Calling BaseAgent init instead of DQN. Weird
        # pylint: disable=bad-super-call
        super(Agent.__bases__[0], self).__init__(env=env, config=config,
                         log_dir=log_dir, tb_writer=tb_writer)

        self.config = config
        self.check_args()

        self.continousActionSpace = False
        if env.action_space.__class__.__name__ == 'Discrete':
            # self.action_space = env.action_space.n * \
            #     len(config.adaptive_repeat)
            ValueError("Discrete Action Spaces are not supported with SAC")
        elif env.action_space.__class__.__name__ == 'Box':
            self.action_space = env.action_space
            self.continousActionSpace = True
        else:
            ValueError('[ERROR] Unrecognized Action Space Type')

        self.num_feats = env.observation_space.shape
        self.envs = env

        self.declare_networks()

        self.policy_optimizer = optim.Adam(self.ac.pi.parameters(), lr=self.config.lr, eps=self.config.adam_eps)
        self.value_optimizer = optim.Adam(self.q_params, lr=self.config.lr, eps=self.config.adam_eps)

        self.value_loss_fun = torch.nn.MSELoss(reduction='none')

        self.declare_memory()
        self.update_count = 0
        self.nstep_buffer = []

        self.first_action = True

        self.training_priors()

    def check_args(self):
        assert(not self.config.double_dqn), "Double DQN is not supported with SAC"
        assert(
            not self.config.recurrent_policy_gradient), "Recurrent Policy is\
                 not supported with SAC"
        assert((len(self.config.adaptive_repeat) == 1) and (self.config.adaptive_repeat[0] == 4)), \
            f"Adaptive Repeat isn't supported in continuous action spaces; it has been \
                changed from its default value to {self.config.adaptive_repeat}"
        assert(not self.config.dueling_dqn), "Dueling DQN is not supported with SAC"
        assert(not self.config.recurrent_policy_gradient), "GRU is not yet supported with SAC"

    def declare_networks(self):
        # self.policy_net = Actor_SAC(self.num_feats, self.action_space, hidden_dim=256, noisy=self.config.noisy_nets, sigma_init=self.config.sigma_init)
        # self.q_net = DQN_SAC(self.num_feats, self.action_space, hidden_dim=256, noisy=self.config.noisy_nets, sigma_init=self.config.sigma_init)
        # self.target_q_net = deepcopy(self.q_net)

        # # First layer of protection. Don't compute gradient for target networks
        # for p in self.target_q_net.parameters():
        #     p.requires_grad = False

        # # move to correct device
        # self.policy_net.to(self.config.device)
        # self.q_net.to(self.config.device)
        # self.target_q_net.to(self.config.device)

        # if self.config.inference:
        #     self.policy_net.eval()
        #     self.q_net.eval()
        #     self.target_q_net.eval()
        # else:
        #     self.policy_net.train()
        #     self.q_net.train()
        #     self.target_q_net.train()
        ac_kwargs = {'hidden_sizes': [256, 256]}
        self.ac = actor_critic(self.envs.observation_space, self.envs.action_space, **ac_kwargs)
        self.ac_targ = deepcopy(self.ac)

        # First layer of protection. Don't compute gradient for target networks
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        
        self.ac.to(self.config.device)
        self.ac_targ.to(self.config.device)

    def declare_memory(self):
        # if self.config.priority_replay:
        #     self.memory = PrioritizedReplayMemory(
        #         self.config.exp_replay_size, self.config.priority_alpha, self.config.priority_beta_start, self.config.priority_beta_tsteps)
        # else:
        #     self.memory = ExperienceReplayMemory(self.config.exp_replay_size)

        self.memory = ReplayBuffer(self.envs.observation_space.shape, self.envs.action_space.shape[0], self.config.exp_replay_size)

    def append_to_replay(self, s, a, r, s_, t):
        # assert(all(s.shape[0] == other.shape[0] for other in (a, r, s_, t))), \
        #     f"First dim of s {s.shape}, a {a.shape}, r {r.shape}, s_ {s_.shape}, t {t.shape} must be equal."

        # for state, action, reward, next_state, terminal in zip(s, a, r, s_, t):
        #     next_state = None if terminal >= 1 else next_state
        #     self.memory.push((state, action, reward, next_state))
        self.memory.store(s, a, r, s_, t)

    def prep_minibatch(self, tstep):
        # # random transition batch is taken from experience replay memory
        # data, indices, weights = self.memory.sample(
        #     self.config.batch_size, tstep)
        # batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values = data

        # batch_state = torch.from_numpy(batch_state).to(
        #     self.config.device).to(torch.float)
        # batch_action = torch.from_numpy(batch_action).to(
        #     self.config.device).to(torch.float)
        # batch_reward = torch.from_numpy(batch_reward).to(
        #     self.config.device).to(torch.float).unsqueeze(dim=1)

        # non_final_mask = torch.from_numpy(non_final_mask).to(
        #     self.config.device).to(torch.bool)
        # if not empty_next_state_values:
        #     non_final_next_states = torch.from_numpy(
        #         non_final_next_states).to(self.config.device).to(torch.float)

        # if self.config.priority_replay:
        #     weights = torch.from_numpy(weights).to(
        #         self.config.device).to(torch.float).view(-1, 1)

        # batch_state /= self.config.state_norm
        # non_final_next_states /= self.config.state_norm

        # return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights

        batch = self.memory.sample_batch(self.config.batch_size)
        for key, value in batch.items():
            batch[key] = value.to(self.config.device)

        return batch

    def compute_value_loss(self, batch_vars, tstep):
        # batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        # # estimate
        # # self.q_net.sample_noise()
        # # current_q_values_1, current_q_values_2 = self.q_net(batch_state, batch_action)
        # current_q_values_1 = self.ac.q1(batch_state, batch_action)
        # current_q_values_2 = self.ac.q2(batch_state, batch_action)

        # # target
        # with torch.no_grad():
        #     next_action_log_probs = torch.zeros(self.config.batch_size, device=self.config.device, dtype=torch.float).unsqueeze(dim=1)
        #     next_q_values_1 = torch.zeros(self.config.batch_size, device=self.config.device, dtype=torch.float).unsqueeze(dim=1)
        #     next_q_values_2 = torch.zeros(self.config.batch_size, device=self.config.device, dtype=torch.float).unsqueeze(dim=1)

        #     # next_actions, next_action_log_probs[non_final_mask] = self.get_action(non_final_next_states, deterministic=False, with_logprob=True)
        #     next_actions, next_action_log_probs[non_final_mask] = self.ac.pi(non_final_next_states)

        #     # self.target_q_net.sample_noise()

        #     if not empty_next_state_values:
        #         # next_q_values_1[non_final_mask], next_q_values_2[non_final_mask] = self.target_q_net(non_final_next_states, next_actions)
        #         next_q_values_1[non_final_mask] = self.ac_targ.q1(non_final_next_states, next_actions)
        #         next_q_values_2[non_final_mask] = self.ac_targ.q2(non_final_next_states, next_actions)
        #         next_q_values = torch.min(next_q_values_1, next_q_values_2)

        #     target = batch_reward + self.config.gamma * (next_q_values - self.config.entropy_coef * next_action_log_probs)

        # loss_q1 = self.value_loss_fun(current_q_values_1, target)
        # loss_q2 = self.value_loss_fun(current_q_values_2, target)

        # if self.config.priority_replay:
        #     with torch.no_grad():
        #         diff = torch.abs(loss_q1 + loss_q2).squeeze().cpu().numpy().tolist()
        #         self.memory.update_priorities(indices, diff)
        #     loss_q1 *= weights
        #     loss_q2 *= weights

        # value_loss = loss_q1.mean() + loss_q2.mean()

        # # Optimize the model
        # self.value_optimizer.zero_grad()
        # value_loss.backward()
        # self.value_optimizer.step()

        # # log val estimates
        # with torch.no_grad():
        #     self.tb_writer.add_scalar('Policy/Value Estimate', torch.cat(
        #         (current_q_values_1, current_q_values_2)).detach().mean().item(), tstep)
        #     self.tb_writer.add_scalar(
        #         'Policy/Next Value Estimate', target.detach().mean().item(), tstep)
        #     self.tb_writer.add_scalar(
        #         'Policy/Entropy Coefficient', self.config.entropy_coef, tstep)
        #     self.tb_writer.add_scalar(
        #         'Loss/Value Loss', value_loss.item(), tstep)

        # return value_loss.item()

        o, a, r, o2, d = batch_vars['obs'], batch_vars['act'], batch_vars['rew'], batch_vars['obs2'], batch_vars['done']


        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.config.gamma * (1 - d) * (q_pi_targ - self.config.entropy_coef * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                      Q2Vals=q2.cpu().detach().numpy())

        return loss_q, q_info

    def compute_policy_loss(self, batch_vars, tstep):
        # batch_state, _, _, _, _, _, _, _ = batch_vars

        # # Compute policy loss
        # # actions, log_probs = self.get_action(batch_state, deterministic=False, with_logprob=True)
        # actions, log_probs = self.ac.pi(batch_state)

        # # q_val1, q_val2 = self.q_net(batch_state, actions)
        # q_val1, q_val2 = self.ac.q1(batch_state, actions), self.ac.q2(batch_state, actions)
        # q_val = torch.min(q_val1, q_val2)

        # policy_loss = (self.config.entropy_coef * log_probs - q_val).mean()

        # # Optimize the model
        # self.policy_optimizer.zero_grad()
        # policy_loss.backward()
        # self.policy_optimizer.step()

        # # log val estimates
        # with torch.no_grad():
        #     self.tb_writer.add_scalar(
        #         'Loss/Policy Loss', policy_loss.item(), tstep)

        # return policy_loss.item()

        o = batch_vars['obs']
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.config.entropy_coef * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.cpu().detach().numpy())

        return loss_pi, pi_info

    def compute_loss(self, batch_vars, tstep):
        # value_loss = self.compute_value_loss(batch_vars, tstep)

        # # turn off gradient for q networks to save compute
        # for p in self.q_params:
        #     p.requires_grad = False

        # policy_loss = self.compute_policy_loss(batch_vars, tstep)

        # for p in self.q_params:
        #     p.requires_grad = True

        # return value_loss + policy_loss

        # First run one gradient descent step for Q1 and Q2
        self.value_optimizer.zero_grad()
        loss_q, q_info = self.compute_value_loss(batch_vars, tstep)
        loss_q.backward()
        self.value_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.policy_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_policy_loss(batch_vars, tstep)
        loss_pi.backward()
        self.policy_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        return 0.0

    def update_(self, tstep=0):
        # TODO: add support for more than one update here
        #   also add support for this to DQN

        if tstep < self.config.learn_start:
            return None

        loss = []
        # For SAC, updates/env_steps == 1
        for _ in range(self.config.update_freq):
            batch_vars = self.prep_minibatch(tstep)
            loss.append(self.compute_loss(batch_vars, tstep))

            self.update_target_model()

        # # more logging
        # with torch.no_grad():
        #     self.tb_writer.add_scalar('Loss/Total Loss', np.mean(loss), tstep)
        #     # self.tb_writer.add_scalar('Learning/Learning Rate', np.mean(
        #     #     [param_group['lr'] for param_group in self.optimizer.param_groups]), tstep)

        #     # log weight norm
        #     weight_norm = 0.
        #     for p in self.ac.parameters():
        #         param_norm = p.data.norm(2)
        #         weight_norm += param_norm.item() ** 2
        #     weight_norm = weight_norm ** (1./2.)
        #     self.tb_writer.add_scalar(
        #         'Learning/Weight Norm', weight_norm, tstep)

        #     # log grad_norm
        #     grad_norm = 0.
        #     for p in self.ac.parameters():
        #         param_norm = p.grad.data.norm(2)
        #         grad_norm += param_norm.item() ** 2
        #     grad_norm = grad_norm ** (1./2.)
        #     self.tb_writer.add_scalar('Learning/Grad Norm', grad_norm, tstep)

        #     # log sigma param norm
        #     if self.config.noisy_nets:
        #         sigma_norm = 0.
        #         for name, p in self.ac.named_parameters():
        #             if p.requires_grad and 'sigma' in name:
        #                 param_norm = p.data.norm(2)
        #                 sigma_norm += param_norm.item() ** 2
        #         sigma_norm = sigma_norm ** (1./2.)
        #         self.tb_writer.add_scalar(
        #             'Policy/Sigma Norm', sigma_norm, tstep)

    def get_action(self, X, deterministic=False):
        return self.ac.act(X, deterministic).reshape(self.envs.action_space.shape)

    def update_target_model(self):
        for target_param, param in zip(self.ac_targ.parameters(), self.ac.parameters()):
            target_param.data.mul_(self.config.polyak_coef)
            target_param.data.add_((1 - self.config.polyak_coef) * param.data)

    def step(self, current_tstep, step=0):
        if current_tstep < self.config.random_act:
            self.actions = self.envs.action_space.sample()
        else:
            # TODO: modifying step in the next line is incosistent with prior code style
            X = torch.from_numpy(self.observations).to(self.config.device).to(
                torch.float).view((-1,)+self.num_feats) / self.config.state_norm
            self.actions = self.get_action(X, deterministic=False)
            # self.actions = self.actions.detach().view(self.envs.action_space.shape).cpu().numpy()

        self.prev_observations = self.observations
        self.observations, self.rewards, self.dones, self.infos = self.envs.step(self.actions)

        self.episode_rewards += self.rewards

        for idx, done in enumerate(self.dones):
            if done:
                self.reset_hx(idx)

                self.tb_writer.add_scalar(
                    'Performance/Agent Reward', self.episode_rewards[idx], current_tstep+idx)
                self.episode_rewards[idx] = 0

        for idx, info in enumerate(self.infos):
            if 'episode' in info.keys():
                self.last_100_rewards.append(info['episode']['r'])
                self.tb_writer.add_scalar(
                    'Performance/Environment Reward', info['episode']['r'], current_tstep+idx)
                self.tb_writer.add_scalar(
                    'Performance/Episode Length', info['episode']['l'], current_tstep+idx)
            
            # ignore time limit done signal in updates
            if self.config.correct_time_limits and 'bad_transition' in info.keys() and info['bad_transition']:
                self.dones[idx] = False

        # self.append_to_replay(self.prev_observations, self.actions.reshape((self.config.num_envs, -1)),
        #                       self.rewards, self.observations, self.dones.astype(int))
        self.append_to_replay(self.prev_observations, self.actions, self.rewards, self.observations, self.dones)

    def update(self, current_tstep):
        self.update_(current_tstep)

    # TODO: Fix saving
    def save_w(self):
        pass

    # TODO: Fix loading
    def load_w(self):
        pass
