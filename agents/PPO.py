import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from agents.A2C import Agent as A2C

from utils import LinearSchedule

class Agent(A2C):
    def __init__(self, env=None, config=None, log_dir='/tmp/gym', tb_writer=None):
        super(Agent, self).__init__(env=env, config=config, log_dir=log_dir, tb_writer=tb_writer)

        self.optimizer = optim.Adam(self.policy_value_net.parameters(), lr=self.config.lr, eps=self.config.adam_eps)
        
        if self.config.anneal_ppo_clip:
            self.anneal_clip_param_fun = LinearSchedule(self.config.ppo_clip_param, 0.0, 1.0, config.max_tsteps)
        else:
            self.anneal_clip_param_fun = LinearSchedule(self.config.ppo_clip_param, None, 1.0, config.max_tsteps)

    def compute_loss(self, sample, next_value, clip_param):
        observations_batch, states_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ = sample

        values, action_log_probs, dist_entropy, states = self.evaluate_actions(observations_batch,
                                                            actions_batch,
                                                            states_batch,
                                                            masks_batch)

        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv_targ
        action_loss = -torch.min(surr1, surr2).mean()

        if self.config.use_ppo_vf_clip:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-clip_param, clip_param)
            value_losses = (values - return_batch).pow(2)
            value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mul(0.5).mean()
        else:
            value_loss = (return_batch - values).pow(2).mul(0.5).mean()

        loss = action_loss + self.config.value_loss_weight * value_loss
        loss -= self.config.entropy_coef * dist_entropy

        return loss, action_loss, value_loss, dist_entropy

    def update_(self, rollout, next_value, tstep):
        rollout.compute_returns(next_value, self.config.gamma)

        advantages = rollout.returns[:-1] - rollout.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        clip_param = self.anneal_clip_param_fun(tstep)

        all_grad_norms = []
        all_sigma_norms = []

        for e in range(self.config.ppo_epoch):
            if self.policy_value_net.use_gru:
                data_generator = rollout.recurrent_generator(
                    advantages, self.config.ppo_mini_batch)
            else:
                data_generator = rollout.feed_forward_generator(
                    advantages, self.config.ppo_mini_batch)


            for sample in data_generator:
                loss, action_loss, value_loss, dist_entropy = self.compute_loss(sample, next_value, clip_param)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_value_net.parameters(), self.config.grad_norm_max)
                self.optimizer.step()

                with torch.no_grad():
                    grad_norm = 0.
                    for p in self.policy_value_net.parameters():
                        param_norm = p.grad.data.norm(2)
                        grad_norm += param_norm.item() ** 2
                    grad_norm = grad_norm ** (1./2.)
                    all_grad_norms.append(grad_norm)

                    if self.config.noisy_nets:
                        sigma_norm = 0.
                        for name, p in self.policy_value_net.named_parameters():
                            if p.requires_grad and 'sigma' in name:
                                param_norm = p.data.norm(2)
                                sigma_norm += param_norm.item() ** 2
                        sigma_norm = sigma_norm ** (1./2.)
                        all_sigma_norms.append(sigma_norm)

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
        
        value_loss_epoch /= (self.config.ppo_epoch * self.config.ppo_mini_batch)
        action_loss_epoch /= (self.config.ppo_epoch * self.config.ppo_mini_batch)
        dist_entropy_epoch /= (self.config.ppo_epoch * self.config.ppo_mini_batch)
        total_loss = value_loss_epoch + action_loss_epoch + dist_entropy_epoch

        self.tb_writer.add_scalar('Loss/Total Loss', total_loss, tstep)
        self.tb_writer.add_scalar('Loss/Policy Loss', action_loss_epoch, tstep)
        self.tb_writer.add_scalar('Loss/Value Loss', value_loss_epoch, tstep)
        self.tb_writer.add_scalar('Loss/Forward Dynamics Loss', 0., tstep)
        self.tb_writer.add_scalar('Loss/Inverse Dynamics Loss', 0., tstep)
        self.tb_writer.add_scalar('Policy/Entropy', dist_entropy_epoch, tstep)
        self.tb_writer.add_scalar('Policy/Value Estimate', 0, tstep)
        if all_sigma_norms:
            self.tb_writer.add_scalar('Policy/Sigma Norm', np.mean(all_sigma_norms), tstep)
        self.tb_writer.add_scalar('Learning/Learning Rate', np.mean([param_group['lr'] for param_group in self.optimizer.param_groups]), tstep)
        self.tb_writer.add_scalar('Learning/Grad Norm', np.mean(all_grad_norms), tstep)

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, 0.
