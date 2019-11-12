import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from agents.A2C import Model as A2C

class Model(A2C):
    def __init__(self, static_policy=False, env=None, config=None, log_dir='/tmp/gym', tb_writer=None):
        super(Model, self).__init__(static_policy, env, config, log_dir, tb_writer=tb_writer)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, eps=self.config.adam_eps)

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
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = 0.5 * (return_batch - values).pow(2).mean()

        loss = action_loss + self.config.value_loss_weight * value_loss
        loss -= self.config.entropy_loss_weight * dist_entropy

        return loss, action_loss, value_loss, dist_entropy

    def update(self, rollout, next_value, frame):
        rollout.compute_returns(next_value, self.config.gamma)

        advantages = rollout.returns[:-1] - rollout.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        if self.config.anneal_ppo_clip:
            clip_param = self.linear_anneal_scalar(self.config.ppo_clip_param, frame, self.config.MAX_FRAMES)
        else:
            clip_param = self.config.ppo_clip_param

        all_grad_norms = []
        all_sigma_norms = []

        for e in range(self.config.ppo_epoch):
            if self.model.use_gru:
                data_generator = rollout.recurrent_generator(
                    advantages, self.config.num_mini_batch)
            else:
                data_generator = rollout.feed_forward_generator(
                    advantages, self.config.num_mini_batch)


            for sample in data_generator:
                loss, action_loss, value_loss, dist_entropy = self.compute_loss(sample, next_value, clip_param)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_max)
                self.optimizer.step()

                with torch.no_grad():
                    grad_norm = 0.
                    for p in self.model.parameters():
                        param_norm = p.grad.data.norm(2)
                        grad_norm += param_norm.item() ** 2
                    grad_norm = grad_norm ** (1./2.)
                    all_grad_norms.append(grad_norm)

                    if self.config.noisy_nets:
                        sigma_norm = 0.
                        for name, p in self.model.named_parameters():
                            if p.requires_grad and 'sigma' in name:
                                param_norm = p.data.norm(2)
                                sigma_norm += param_norm.item() ** 2
                        sigma_norm = sigma_norm ** (1./2.)
                        all_sigma_norms.append(sigma_norm)

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
        
        value_loss_epoch /= (self.config.ppo_epoch * self.config.num_mini_batch)
        action_loss_epoch /= (self.config.ppo_epoch * self.config.num_mini_batch)
        dist_entropy_epoch /= (self.config.ppo_epoch * self.config.num_mini_batch)
        total_loss = value_loss_epoch + action_loss_epoch + dist_entropy_epoch

        self.tb_writer.add_scalar('Loss/Total Loss', total_loss, frame)
        self.tb_writer.add_scalar('Loss/Policy Loss', action_loss_epoch, frame)
        self.tb_writer.add_scalar('Loss/Value Loss', value_loss_epoch, frame)
        self.tb_writer.add_scalar('Loss/Forward Dynamics Loss', 0., frame)
        self.tb_writer.add_scalar('Loss/Inverse Dynamics Loss', 0., frame)
        self.tb_writer.add_scalar('Policy/Entropy', dist_entropy_epoch, frame)
        self.tb_writer.add_scalar('Policy/Value Estimate', 0, frame)
        self.tb_writer.add_scalar('Policy/Sigma Norm', np.mean(all_sigma_norms), frame)
        self.tb_writer.add_scalar('Learning/Learning Rate', np.mean([param_group['lr'] for param_group in self.optimizer.param_groups]), frame)
        self.tb_writer.add_scalar('Learning/Grad Norm', np.mean(all_grad_norms), frame)

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, 0.
    
    def linear_anneal_scalar(self, initial_val, frame, max_frames):
        val = initial_val - (initial_val * (frame / float(max_frames)))
        return val
