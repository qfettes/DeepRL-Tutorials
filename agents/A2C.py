import numpy as np

import os

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from agents.BaseAgent import BaseAgent
from networks.networks import ActorCritic
from utils.RolloutStorage import RolloutStorage

from collections import deque

class Agent(BaseAgent):
    def __init__(self, env=None, config=None, log_dir='/tmp/gym', tb_writer=None):
        super(Agent, self).__init__(env=env, config=config, log_dir=log_dir, tb_writer=tb_writer)
        self.config = config
        self.num_feats = env.observation_space.shape

        self.continousActionSpace = False
        if env.action_space.__class__.__name__ == 'Discrete':
            self.action_space = env.action_space.n * len(config.adaptive_repeat)
        elif env.action_space.__class__.__name__ == 'Box':
            self.action_space = env.action_space
            self.continousActionSpace = True
        else:
            print('[ERROR] Unrecognized Action Space Type')
            exit()
        
        self.envs = env

        self.declare_networks()

        self.optimizer = optim.RMSprop(self.q_func.parameters(), lr=self.config.lr, alpha=self.config.rms_alpha, eps=self.config.rms_eps)   
        
        #move to correct device
        self.q_func = self.q_func.to(self.config.device)

        if self.config.inference:
            self.q_func.eval()
        else:
            self.q_func.train()

        self.rollouts = RolloutStorage(self.config.update_freq , self.config.num_envs,
            self.num_feats, self.envs.action_space, self.q_func.state_size,
            self.config.device, config.use_gae, config.gae_tau, config.correct_time_limits)

        self.value_losses = []
        self.entropy_losses = []
        self.policy_losses = []
        
        self.first_action = True

        self.training_priors()


    def declare_networks(self):
        self.q_func = ActorCritic(self.num_feats, self.action_space, body_out=self.config.body_out, use_gru=self.config.policy_gradient_recurrent_policy, gru_size=self.config.gru_size, noisy_nets=self.config.noisy_nets, sigma_init=self.config.sigma_init)

    def training_priors(self):
        self.obs = self.envs.reset()
    
        obs = torch.from_numpy(self.obs.astype(np.float32)).to(self.config.device)

        self.rollouts.observations[0].copy_(obs)
        
        self.episode_rewards = np.zeros(self.config.num_envs, dtype=np.float)
        self.final_rewards = np.zeros(self.config.num_envs, dtype=np.float)
        self.last_100_rewards = deque(maxlen=100)

    def get_action(self, s, states, masks, deterministic=False):
        if self.first_action:
            self.add_graph(s, states, masks)

        logits, values, states = self.q_func(s, states, masks)

        #TODO: clean this up
        if self.continousActionSpace:
            dist = torch.distributions.Normal(logits, F.softplus(self.q_func.logstd))
            actions = dist.sample()
            action_log_probs = dist.log_prob(actions).sum(-1, keepdim=True)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            if deterministic:
                #TODO: different in original
                actions = dist.probs.argmax(dim=1, keepdim=True)
            else:
                actions = dist.sample().view(-1, 1)

            log_probs = F.log_softmax(logits, dim=1)
            action_log_probs = log_probs.gather(1, actions)

        return values, actions, action_log_probs, states

    def evaluate_actions(self, s, actions, states, masks):
        logits, values, states = self.q_func(s, states, masks)

        #TODO: clean this up
        if self.continousActionSpace:
            dist = torch.distributions.Normal(logits, F.softplus(self.q_func.logstd))
            action_log_probs = dist.log_prob(actions).sum(-1, keepdim=True)
            dist_entropy = dist.entropy().sum(1).mean()
        else:
            dist = torch.distributions.Categorical(logits=logits)

            log_probs = F.log_softmax(logits, dim=1)
            action_log_probs = log_probs.gather(1, actions)

            dist_entropy = dist.entropy().mean()

        return values, action_log_probs, dist_entropy, states

    def get_values(self, s, states, masks):
        _, values, _ = self.q_func(s, states, masks)

        return values

    def compute_loss(self, rollouts, next_value, tstep):
        obs_shape = rollouts.observations.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        rollouts.compute_returns(next_value, self.config.gamma)

        values, action_log_probs, dist_entropy, states = self.evaluate_actions(
            rollouts.observations[:-1].view(-1, *obs_shape),
            rollouts.actions.view(-1, rollouts.actions.shape[-1]),
            rollouts.states[0].view(-1, self.q_func.state_size),
            rollouts.masks[:-1].view(-1, 1))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mul(0.5).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        loss = action_loss + self.config.value_loss_weight * value_loss
        loss -= self.config.entropy_loss_weight * dist_entropy

        self.tb_writer.add_scalar('Loss/Total Loss', loss.item(), tstep)
        self.tb_writer.add_scalar('Loss/Policy Loss', action_loss.item(), tstep)
        self.tb_writer.add_scalar('Loss/Value Loss', value_loss.item(), tstep)
        self.tb_writer.add_scalar('Loss/Forward Dynamics Loss', 0., tstep)
        self.tb_writer.add_scalar('Loss/Inverse Dynamics Loss', 0., tstep)

        self.tb_writer.add_scalar('Policy/Entropy', dist_entropy.item(), tstep)
        self.tb_writer.add_scalar('Policy/Value Estimate', values.detach().mean().item(), tstep)

        self.tb_writer.add_scalar('Learning/Learning Rate', np.mean([param_group['lr'] for param_group in self.optimizer.param_groups]), tstep)

        return loss, action_loss, value_loss, dist_entropy, 0.

    def update_(self, rollout, next_value, tstep):
        loss, action_loss, value_loss, dist_entropy, dynamics_loss = self.compute_loss(rollout, next_value, tstep)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_func.parameters(), self.config.grad_norm_max)
        self.optimizer.step()

        with torch.no_grad():
            grad_norm = 0.
            for p in self.q_func.parameters():
                if p.requires_grad:
                    param_norm = p.grad.data.norm(2)
                    grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm ** (1./2.)

            self.tb_writer.add_scalar('Learning/Grad Norm', grad_norm, tstep)

            if self.config.noisy_nets:
                sigma_norm = 0.
                for name, p in self.q_func.named_parameters():
                    if p.requires_grad and 'sigma' in name:
                        param_norm = p.data.norm(2)
                        sigma_norm += param_norm.item() ** 2
                sigma_norm = sigma_norm ** (1./2.)

                self.tb_writer.add_scalar('Policy/Sigma Norm', sigma_norm, tstep)

        return value_loss.item(), action_loss.item(), dist_entropy.item(), dynamics_loss

    def add_graph(self, inp, states, masks):
        with torch.no_grad():
            self.tb_writer.add_graph(self.q_func, (inp, states, masks))
            self.first_action = False
    
    def step(self, current_timestep, step=0):
        with torch.no_grad():
            values, actions, action_log_prob, states = self.get_action(
                                                        self.rollouts.observations[step],
                                                        self.rollouts.states[step],
                                                        self.rollouts.masks[step])
    
        cpu_actions = actions.cpu().numpy()
        if not self.continousActionSpace:
            cpu_actions = cpu_actions.reshape(-1)

        obs, reward, done, info = self.envs.step(cpu_actions)

        obs = torch.from_numpy(obs.astype(np.float32)).to(self.config.device)

        #agent rewards
        self.episode_rewards += reward
        masks = 1. - done.astype(np.float32)
        self.final_rewards *= masks
        self.final_rewards += (1. - masks) * self.episode_rewards
        self.episode_rewards *= masks

        bad_masks = []
        for idx, inf in enumerate(info):
            if 'episode' in inf.keys():
                self.last_100_rewards.append(inf['episode']['r'])
                self.tb_writer.add_scalar('Performance/Environment Reward', inf['episode']['r'], current_timestep+idx)
                self.tb_writer.add_scalar('Performance/Episode Length', inf['episode']['l'], current_timestep+idx)

            if done[idx]:
                #write reward on completion
                self.tb_writer.add_scalar('Performance/Agent Reward', self.final_rewards[idx], current_timestep+idx)

            if 'bad_transition' in inf.keys():
                bad_masks.append([0.0])
            else:
                bad_masks.append([1.0])

        rewards = torch.from_numpy(reward.astype(np.float32)).view(-1, 1).to(self.config.device)
        masks = torch.from_numpy(masks).to(self.config.device).view(-1, 1)
        bad_masks = torch.tensor(bad_masks, dtype=torch.float, device=self.config.device)

        self.rollouts.insert(obs, states, actions, action_log_prob, values, rewards, masks, bad_masks)

    def update(self, current_tstep):
        with torch.no_grad():
            next_value = self.get_values(self.rollouts.observations[-1],
                                self.rollouts.states[-1],
                                self.rollouts.masks[-1])
        
        if current_tstep >= self.config.learn_start:
            value_loss, action_loss, dist_entropy, dynamics_loss = self.update_(self.rollouts, next_value, current_tstep)
        
        self.rollouts.after_update()
