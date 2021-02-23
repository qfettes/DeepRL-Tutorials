from __future__ import absolute_import

import itertools
import os
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
from networks.networks import DQN_SAC, Actor_SAC
# from networks.networks import MLPActorCritic as actor_critic
from utils.ReplayMemory import ExperienceReplayMemory, PrioritizedReplayMemory

from agents.BaseAgent import BaseAgent
from agents.DQN import Agent as DQN_Agent


class Agent(DQN_Agent):
    def __init__(self, env=None, config=None, log_dir='/tmp/gym', tb_writer=None,
        valid_arguments=set(), default_arguments={}):
        # NOTE: Calling BaseAgent init instead of DQN. Weird
        # pylint: disable=bad-super-call
        super(Agent.__bases__[0], self).__init__(env=env, config=config,
                         log_dir=log_dir, tb_writer=tb_writer,
                         valid_arguments=valid_arguments,
                         default_arguments=default_arguments)

        self.config = config

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

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.lr, eps=self.config.optim_eps)
        self.value_optimizer = optim.Adam(self.q_net.parameters(), lr=self.config.lr, eps=self.config.optim_eps)

        if self.config.entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(self.device)).item()
            self.log_entropy_coef = torch.zeros(1, requires_grad=True, device=self.device)
            self.config.entropy_coef = self.log_entropy_coef.exp()
            self.entropy_optimizer = optim.Adam([self.log_entropy_coef], lr=self.config.lr)

        self.value_loss_fun = torch.nn.MSELoss(reduction='none')

        self.declare_memory()
        self.update_count = 0
        self.nstep_buffer = []

        self.training_priors()    

    def declare_networks(self):
        self.policy_net = Actor_SAC(self.num_feats, self.action_space, hidden_dim=256, noisy_nets=self.config.noisy_nets, noisy_sigma=self.config.noisy_sigma)
        self.q_net = DQN_SAC(self.num_feats, self.action_space, hidden_dim=256, noisy_nets=self.config.noisy_nets, noisy_sigma=self.config.noisy_sigma)
        self.target_q_net = deepcopy(self.q_net)

        # First layer of protection. Don't compute gradient for target networks
        for p in self.target_q_net.parameters():
            p.requires_grad = False

        # move to correct device
        self.policy_net.to(self.device)
        self.q_net.to(self.device)
        self.target_q_net.to(self.device)

        if self.config.inference:
            self.policy_net.eval()
            self.q_net.eval()
            self.target_q_net.eval()
        else:
            self.policy_net.train()
            self.q_net.train()
            self.target_q_net.train()

        self.all_named_params = itertools.chain(self.policy_net.named_parameters(), self.q_net.named_parameters())

    def compute_value_loss(self, batch_vars, tstep):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        # estimate
        self.q_net.sample_noise()
        current_q_values_1, current_q_values_2 = self.q_net(batch_state, batch_action)

        # target
        with torch.no_grad():
            next_action_log_probs = torch.zeros(self.config.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            next_q_values_1 = torch.zeros(self.config.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            next_q_values_2 = torch.zeros(self.config.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)


            self.target_q_net.sample_noise()
            next_actions, next_action_log_probs[non_final_mask] = self.policy_net(non_final_next_states)

            if not empty_next_state_values:
                next_q_values_1[non_final_mask], next_q_values_2[non_final_mask] = self.target_q_net(non_final_next_states, next_actions)
                next_q_values = torch.min(next_q_values_1, next_q_values_2)

            target = batch_reward + self.config.gamma * (next_q_values - self.config.entropy_coef * next_action_log_probs)

        loss_q1 = self.value_loss_fun(current_q_values_1, target)
        loss_q2 = self.value_loss_fun(current_q_values_2, target)

        if self.config.priority_replay:
            with torch.no_grad():
                diff = torch.abs(loss_q1 + loss_q2).squeeze().cpu().numpy().tolist()
                self.memory.update_priorities(indices, diff)
            loss_q1 *= weights
            loss_q2 *= weights

        value_loss = loss_q1.mean() + loss_q2.mean()

        # log val estimates
        if self.tb_writer:
            with torch.no_grad():
                self.tb_writer.add_scalar('Policy/Value Estimate', torch.cat(
                    (current_q_values_1, current_q_values_2)).detach().mean().item(), tstep)
                self.tb_writer.add_scalar(
                    'Policy/Next Value Estimate', target.detach().mean().item(), tstep)
                self.tb_writer.add_scalar(
                    'Policy/Entropy Coefficient', self.config.entropy_coef, tstep)
                self.tb_writer.add_scalar(
                    'Loss/Value Loss', value_loss.detach().item(), tstep)

        return value_loss

    def compute_policy_loss(self, batch_vars, tstep):
        batch_state, _, _, _, _, _, _, _ = batch_vars

        # Compute policy loss
        self.policy_net.sample_noise()
        actions, log_probs = self.policy_net(batch_state)

        self.q_net.sample_noise()
        q_val1, q_val2 = self.q_net(batch_state, actions)
        q_val = torch.min(q_val1, q_val2)

        policy_loss = (self.config.entropy_coef * log_probs - q_val).mean()

        # log val estimates
        if self.tb_writer:
            with torch.no_grad():
                self.tb_writer.add_scalar(
                    'Loss/Policy Loss', policy_loss.detach().item(), tstep)

        return policy_loss, log_probs

    def compute_entropy_loss(self, action_log_probs, tstep):
        entropy_loss = -(self.log_entropy_coef * (action_log_probs + self.target_entropy).detach()).mean()

        if self.tb_writer:
            with torch.no_grad():
                self.tb_writer.add_scalar(
                    'Loss/Policy Loss', entropy_loss.detach().item(), tstep)

        return entropy_loss

    def compute_loss(self, batch_vars, tstep):
        # First run one gradient descent step for Q1 and Q2
        self.value_optimizer.zero_grad()
        loss_q = self.compute_value_loss(batch_vars, tstep)
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q_net.parameters(), self.config.grad_norm_max)
        self.value_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_net.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.policy_optimizer.zero_grad()
        loss_pi, action_log_probs = self.compute_policy_loss(batch_vars, tstep)
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), self.config.grad_norm_max)
        self.policy_optimizer.step()

        loss_entropy = torch.zeros((1,)).to(self.device)
        if self.config.entropy_tuning:
            self.entropy_optimizer.zero_grad()
            loss_entropy = self.compute_entropy_loss(action_log_probs, tstep)
            loss_entropy.backward()
            torch.nn.utils.clip_grad_norm_(
                [self.log_entropy_coef], self.config.grad_norm_max)
            self.entropy_optimizer.step()

            self.config.entropy_coef = self.log_entropy_coef.exp().detach()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_net.parameters():
            p.requires_grad = True

        return loss_pi.detach().cpu().item() + loss_q.detach().cpu().item() + loss_entropy.detach().cpu().item()

    def update_(self, tstep=0):
        loss = []
        # For SAC, updates/env_steps == 1
        for _ in range(self.config.update_freq):
            batch_vars = self.prep_minibatch(tstep)
            loss.append(self.compute_loss(batch_vars, tstep))

            self.update_target_model()

        # more logging
        if self.tb_writer:
            with torch.no_grad():
                self.tb_writer.add_scalar('Loss/Total Loss', np.mean(loss), tstep)
                self.tb_writer.add_scalar('Learning/Policy Learning Rate', np.mean([param_group['lr'] for param_group in self.policy_optimizer.param_groups]), tstep)
                self.tb_writer.add_scalar('Learning/Value Learning Rate', np.mean([param_group['lr'] for param_group in self.value_optimizer.param_groups]), tstep)

                # log weight norm
                weight_norm = 0.
                for _, p in self.all_named_params:
                    param_norm = p.data.norm(2)
                    weight_norm += param_norm.item() ** 2
                weight_norm = weight_norm ** (1./2.)
                self.tb_writer.add_scalar(
                    'Learning/Weight Norm', weight_norm, tstep)

                # log grad_norm
                grad_norm = 0.
                for _, p in self.all_named_params:
                    param_norm = p.grad.data.norm(2)
                    grad_norm += param_norm.item() ** 2
                grad_norm = grad_norm ** (1./2.)
                self.tb_writer.add_scalar('Learning/Grad Norm', grad_norm, tstep)

                # log sigma param norm
                if self.config.noisy_nets:
                    sigma_norm = 0.
                    for name, p in self.all_named_params:
                        if p.requires_grad and 'sigma' in name:
                            param_norm = p.data.norm(2)
                            sigma_norm += param_norm.item() ** 2
                    sigma_norm = sigma_norm ** (1./2.)
                    self.tb_writer.add_scalar(
                        'Policy/Sigma Norm', sigma_norm, tstep)

    def get_action(self, obs, deterministic=False):
        self.policy_net.sample_noise()
        X = torch.from_numpy(obs).to(self.device).to(torch.float).view((-1,)+self.num_feats) / self.config.state_norm
        return self.policy_net.act(X, deterministic).reshape(self.envs.action_space.shape)

    def update_target_model(self):
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.mul_(self.config.polyak_coef)
            target_param.data.add_((1 - self.config.polyak_coef) * param.data)

    def step(self, current_tstep, step=0):
        if current_tstep < self.config.random_act:
            self.actions = self.envs.action_space.sample()
        else:
            self.actions = self.get_action(self.observations, deterministic=self.config.inference)

        self.prev_observations = self.observations
        self.observations, self.rewards, self.dones, self.infos = self.envs.step(self.actions)

        self.episode_rewards += self.rewards

        for idx, done in enumerate(self.dones):
            if done:
                self.reset_hx(idx)

                if self.tb_writer:
                    self.tb_writer.add_scalar(
                        'Performance/Agent Reward', self.episode_rewards[idx], current_tstep+idx)
                self.episode_rewards[idx] = 0

        for idx, info in enumerate(self.infos):
            if 'episode' in info.keys():
                self.last_100_rewards.append(info['episode']['r'])
                if self.tb_writer:
                    self.tb_writer.add_scalar(
                        'Performance/Environment Reward', info['episode']['r'], current_tstep+idx)
                    self.tb_writer.add_scalar(
                        'Performance/Episode Length', info['episode']['l'], current_tstep+idx)
            
            # ignore time limit done signal in updates
            if self.config.correct_time_limits and 'bad_transition' in info.keys() and info['bad_transition']:
                self.dones[idx] = False

        self.append_to_replay(self.prev_observations, self.actions.reshape((self.config.nenvs, -1)),
                              self.rewards, self.observations, self.dones.astype(int))

    def update(self, current_tstep):
        self.update_(current_tstep)

    def save_w(self):
        torch.save(self.policy_net.state_dict(), os.path.join(
            self.log_dir, 'saved_model', 'policy_model.dump'))
        torch.save(self.policy_optimizer.state_dict(), os.path.join(
            self.log_dir, 'saved_model', 'policy_optim.dump'))

        torch.save(self.q_net.state_dict(), os.path.join(
            self.log_dir, 'saved_model', 'value_model.dump'))
        torch.save(self.value_optimizer.state_dict(), os.path.join(
            self.log_dir, 'saved_model', 'value_optim.dump'))

        if self.config.entropy_tuning:
            torch.save(self.log_entropy_coef, os.path.join(
            self.log_dir, 'saved_model', 'entropy_model.dump'))
            torch.save(self.entropy_optimizer, os.path.join(
                self.log_dir, 'saved_model', 'entropy_optim.dump'))

    def load_w(self):
        fname_model = os.path.join(self.log_dir, 'saved_model', 'policy_model.dump')
        fname_optim = os.path.join(self.log_dir, 'saved_model', 'policy_optim.dump')
        if os.path.isfile(fname_model):
            self.policy_net.load_state_dict(torch.load(fname_model))
        if os.path.isfile(fname_optim):
            self.policy_optimizer.load_state_dict(torch.load(fname_optim))

        fname_model = os.path.join(self.log_dir, 'saved_model', 'value_model.dump')
        fname_optim = os.path.join(self.log_dir, 'saved_model', 'value_optim.dump')
        if os.path.isfile(fname_model):
            self.q_net.load_state_dict(torch.load(fname_model))
            self.target_q_net = deepcopy(self.q_net)
        if os.path.isfile(fname_optim):
            print(torch.load(fname_optim))
            self.value_optimizer.load_state_dict(torch.load(fname_optim))

        if self.config.entropy_tuning:
            fname_model = os.path.join(self.log_dir, 'saved_model', 'entropy_model.dump')
            fname_optim = os.path.join(self.log_dir, 'saved_model', 'entropy_optim.dump')
            if os.path.isfile(fname_model):
                self.log_entropy_coef = torch.load(fname_model)
            if os.path.isfile(fname_optim): # not exactly right
                self.entropy_optimizer = optim.Adam([self.log_entropy_coef], lr=self.config.lr)
