from __future__ import absolute_import

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from networks.layers import NoisyLinear
from networks.network_bodies import (AtariBody, AtariBodyAC, SimpleBody,
                                     SimpleBodyAC)

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions, noisy_nets=False, noisy_sigma=0.5, body=SimpleBody):
        super().__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.noisy = noisy_nets

        self.body = body(input_shape, num_actions, noisy, noisy_sigma)

        self.fc1 = nn.Linear(self.body.feature_size(), 512) if not self.noisy else NoisyLinear(
            self.body.feature_size(), 512, noisy_sigma)
        self.fc2 = nn.Linear(512, self.num_actions) if not self.noisy else NoisyLinear(
            512, self.num_actions, noisy_sigma)

    def forward(self, x):
        x = self.body(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.fc1.sample_noise()
            self.fc2.sample_noise()


class DuelingDQN(nn.Module):
    def __init__(self, input_shape, num_outputs, noisy_nets=False, noisy_sigma=0.5, body=SimpleBody):
        super().__init__()

        self.input_shape = input_shape
        self.num_actions = num_outputs
        self.noisy = noisy_nets

        self.body = body(input_shape, num_outputs, noisy, noisy_sigma)

        self.adv1 = nn.Linear(self.body.feature_size(), 512) if not self.noisy else NoisyLinear(
            self.body.feature_size(), 512, noisy_sigma)
        self.adv2 = nn.Linear(512, self.num_actions) if not self.noisy else NoisyLinear(
            512, self.num_actions, noisy_sigma)

        self.val1 = nn.Linear(self.body.feature_size(), 512) if not self.noisy else NoisyLinear(
            self.body.feature_size(), 512, noisy_sigma)
        self.val2 = nn.Linear(512, 1) if not self.noisy else NoisyLinear(
            512, 1, noisy_sigma)

    def forward(self, x):
        x = self.body(x)

        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv)

        val = F.relu(self.val1(x))
        val = self.val2(val)

        return val + adv - adv.mean()

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.adv1.sample_noise()
            self.adv2.sample_noise()
            self.val1.sample_noise()
            self.val2.sample_noise()


class CategoricalDQN(nn.Module):
    def __init__(self, input_shape, num_outputs, noisy_nets=False, noisy_sigma=0.5, body=SimpleBody, atoms=51):
        super().__init__()

        self.input_shape = input_shape
        self.num_actions = num_outputs
        self.noisy = noisy_nets
        self.c51_atoms = atoms

        self.body = body(input_shape, num_outputs, noisy, noisy_sigma)

        self.fc1 = nn.Linear(self.body.feature_size(), 512) if not self.noisy else NoisyLinear(
            self.body.feature_size(), 512, noisy_sigma)
        self.fc2 = nn.Linear(512, self.num_actions*self.c51_atoms) if not self.noisy else NoisyLinear(
            512, self.num_actions*self.c51_atoms, noisy_sigma)

    def forward(self, x):
        x = self.body(x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, self.num_actions, self.c51_atoms)

        return F.softmax(x, dim=-1), F.log_softmax(x, dim=-1)

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.fc1.sample_noise()
            self.fc2.sample_noise()


class CategoricalDuelingDQN(nn.Module):
    def __init__(self, input_shape, num_outputs, noisy_nets=False, noisy_sigma=0.5, body=SimpleBody, atoms=51):
        super().__init__()

        self.input_shape = input_shape
        self.num_actions = num_outputs
        self.noisy = noisy_nets
        self.c51_atoms = atoms

        self.body = body(input_shape, num_outputs, noisy, noisy_sigma)

        self.adv1 = nn.Linear(self.body.feature_size(), 512) if not self.noisy else NoisyLinear(
            self.body.feature_size(), 512, noisy_sigma)
        self.adv2 = nn.Linear(512, self.num_actions*self.c51_atoms) if not self.noisy else NoisyLinear(
            512, self.num_actions*self.c51_atoms, noisy_sigma)

        self.val1 = nn.Linear(self.body.feature_size(), 512) if not self.noisy else NoisyLinear(
            self.body.feature_size(), 512, noisy_sigma)
        self.val2 = nn.Linear(
            512, 1*self.c51_atoms) if not self.noisy else NoisyLinear(512, 1*self.c51_atoms, noisy_sigma)

    def forward(self, x):
        x = self.body(x)

        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv).view(-1, self.num_actions, self.c51_atoms)

        val = F.relu(self.val1(x))
        val = self.val2(val).view(-1, 1, self.c51_atoms)

        final = val + adv - adv.mean(dim=1).view(-1, 1, self.c51_atoms)

        return F.softmax(final, dim=-1), F.log_softmax(final, dim=-1)

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.adv1.sample_noise()
            self.adv2.sample_noise()
            self.val1.sample_noise()
            self.val2.sample_noise()


class QRDQN(nn.Module):
    def __init__(self, input_shape, num_outputs, noisy_nets=False, noisy_sigma=0.5, body=SimpleBody, quantiles=51):
        super().__init__()

        self.input_shape = input_shape
        self.num_actions = num_outputs
        self.noisy = noisy_nets
        self.quantiles = quantiles

        self.body = body(input_shape, num_outputs, noisy, noisy_sigma)

        self.fc1 = nn.Linear(self.body.feature_size(), 512) if not self.noisy else NoisyLinear(
            self.body.feature_size(), 512, noisy_sigma)
        self.fc2 = nn.Linear(512, self.num_actions*self.quantiles) if not self.noisy else NoisyLinear(
            512, self.num_actions*self.quantiles, noisy_sigma)

    def forward(self, x):
        x = self.body(x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x.view(-1, self.num_actions, self.quantiles)

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.fc1.sample_noise()
            self.fc2.sample_noise()


class DuelingQRDQN(nn.Module):
    def __init__(self, input_shape, num_outputs, noisy_nets=False, noisy_sigma=0.5, body=SimpleBody, quantiles=51):
        super().__init__()

        self.input_shape = input_shape
        self.num_actions = num_outputs
        self.noisy = noisy_nets
        self.quantiles = quantiles

        self.body = body(input_shape, num_outputs, noisy, noisy_sigma)

        self.adv1 = nn.Linear(self.body.feature_size(), 512) if not self.noisy else NoisyLinear(
            self.body.feature_size(), 512, noisy_sigma)
        self.adv2 = nn.Linear(512, self.num_actions*self.quantiles) if not self.noisy else NoisyLinear(
            512, self.num_actions*self.quantiles, noisy_sigma)

        self.val1 = nn.Linear(self.body.feature_size(), 512) if not self.noisy else NoisyLinear(
            self.body.feature_size(), 512, noisy_sigma)
        self.val2 = nn.Linear(
            512, 1*self.quantiles) if not self.noisy else NoisyLinear(512, 1*self.quantiles, noisy_sigma)

    def forward(self, x):
        x = self.body(x)

        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv).view(-1, self.num_actions, self.quantiles)

        val = F.relu(self.val1(x))
        val = self.val2(val).view(-1, 1, self.quantiles)

        final = val + adv - adv.mean(dim=1).view(-1, 1, self.quantiles)

        return final

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.adv1.sample_noise()
            self.adv2.sample_noise()
            self.val1.sample_noise()
            self.val2.sample_noise()


########Recurrent Architectures#########

class DRQN(nn.Module):
    def __init__(self, input_shape, num_actions, noisy_nets=False, noisy_sigma=0.5, gru_size=512, bidirectional=False, body=SimpleBody):
        super().__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.noisy = noisy_nets
        self.gru_size = gru_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        self.body = body(input_shape, num_actions,
                         noisy_nets=self.noisy, noisy_sigma=noisy_sigma)
        self.gru = nn.GRU(self.body.feature_size(), self.gru_size,
                          num_layers=1, batch_first=True, bidirectional=bidirectional)
        self.fc2 = nn.Linear(self.gru_size, self.num_actions) if not self.noisy else NoisyLinear(
            self.gru_size, self.num_actions, noisy_sigma)

    def forward(self, x, hx=None):
        batch_size = x.size(0)
        sequence_length = x.size(1)

        x = x.view((-1,)+self.input_shape)

        # format outp for batch first gru
        feats = self.body(x).view(batch_size, sequence_length, -1)
        hidden = self.init_hidden(batch_size) if hx is None else hx
        out, hidden = self.gru(feats, hidden)
        x = self.fc2(out)

        return x, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1*self.num_directions, batch_size, self.gru_size, device=self.device, dtype=torch.float)

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.fc2.sample_noise()


########Actor Critic Architectures#########

class ActorCritic(nn.Module):
    def __init__(self, input_shape, num_actions, body_out=64, use_gru=False, gru_size=256, noisy_nets=False, noisy_sigma=0.5):
        super().__init__()
        self.body_out = body_out
        self.use_gru = use_gru
        self.gru_size = gru_size
        self.noisy = noisy_nets

        self.continuous = (num_actions.__class__.__name__ == 'Box')

        if not self.continuous:
            self.body = AtariBodyAC(
                input_shape, body_out, self.noisy, noisy_sigma)
            num_outputs = num_actions
        else:
            self.body = SimpleBodyAC(
                input_shape, body_out, self.noisy, noisy_sigma)
            num_outputs = num_actions.shape[0]
            self.logstd = nn.Parameter(torch.zeros(num_outputs))

        encoder_out = self.gru_size

        def init_(m): return self.layer_init(m, nn.init.orthogonal_,
                                             lambda x: nn.init.constant_(x, 0),
                                             nn.init.calculate_gain('relu'),
                                             noisy_layer=self.noisy)
        if use_gru:
            self.gru = nn.GRU(self.body_out, self.gru_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)
        else:
            if self.continuous:
                self.fc1 = init_(nn.Linear(body_out, self.gru_size)) if not self.noisy else init_(
                    NoisyLinear(body_out, self.gru_size, noisy_sigma))
            else:
                encoder_out = self.body_out

        # final actor and critic layers
        def init_(m): return self.layer_init(m, nn.init.orthogonal_,
                                             lambda x: nn.init.constant_(x, 0), gain=1,
                                             noisy_layer=self.noisy)

        self.critic_linear = init_(nn.Linear(encoder_out, 1)) if not self.noisy else init_(
            NoisyLinear(self.gru_size, 1, noisy_sigma))

        def init_(m): return self.layer_init(m, nn.init.orthogonal_,
                                             lambda x: nn.init.constant_(x, 0), gain=0.01,
                                             noisy_layer=self.noisy)

        self.actor_linear = init_(nn.Linear(encoder_out, num_outputs)) if not self.noisy else init_(
            NoisyLinear(self.gru_size, num_outputs, noisy_sigma))

        self.train()
        if self.noisy:
            self.sample_noise()

    def forward(self, inputs, states, masks):
        x = self.body(inputs)

        if self.use_gru:
            if inputs.size(0) == states.size(0):
                x, states = self.gru(x.unsqueeze(
                    0), (states * masks).unsqueeze(0))
                x = x.squeeze(0)
                states = states.squeeze(0)
            else:
                # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
                N = states.size(0)
                T = int(x.size(0) / N)

                # unflatten
                x = x.view(T, N, x.size(1))

                # Same deal with masks
                masks = masks.view(T, N)

                # Let's figure out which steps in the sequence have a zero for any agent
                # We will always assume t=0 has a zero in it as that makes the logic cleaner
                has_zeros = ((masks[1:] == 0.0)
                             .any(dim=-1)
                             .nonzero()
                             .squeeze()
                             .cpu())

                # +1 to correct the masks[1:]
                if has_zeros.dim() == 0:
                    # Deal with scalar
                    has_zeros = [has_zeros.item() + 1]
                else:
                    has_zeros = (has_zeros + 1).numpy().tolist()

                # add t=0 and t=T to the list
                has_zeros = [0] + has_zeros + [T]

                states = states.unsqueeze(0)
                outputs = []
                for i in range(len(has_zeros) - 1):
                    # We can now process steps that don't have any zeros in masks together!
                    # This is much faster
                    start_idx = has_zeros[i]
                    end_idx = has_zeros[i + 1]

                    rnn_scores, states = self.gru(
                        x[start_idx:end_idx], states * masks[start_idx].view(1, -1, 1))

                    outputs.append(rnn_scores)

                # assert len(outputs) == T
                # x is a (T, N, -1) tensor
                x = torch.cat(outputs, dim=0)

                # flatten
                x = x.view(T * N, -1)
                states = states.squeeze(0)
        elif self.continuous:
            x = self.fc1(x)

        value = self.critic_linear(x)
        logits = self.actor_linear(x)

        return logits, value, states

    def layer_init(self, module, weight_init, bias_init, gain=1, noisy_layer=False):
        if not noisy_layer:
            weight_init(module.weight.data, gain=gain)
            bias_init(module.bias.data)
        else:
            weight_init(module.weight_mu.data, gain=gain)
            bias_init(module.bias_mu.data)
        return module

    def sample_noise(self):
        if self.noisy:
            self.critic_linear.sample_noise()
            self.actor_linear.sample_noise()
            self.fc1.sample_noise()

    @property
    def state_size(self):
        if self.use_gru:
            return self.gru_size
        else:
            return 1


LOG_STD_MAX = 2
LOG_STD_MIN = -20

# def mlp(sizes, activation, output_activation=nn.Identity):
#     layers = []
#     for j in range(len(sizes)-1):
#         act = activation if j < len(sizes)-2 else output_activation
#         layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
#     return nn.Sequential(*layers)

# class SquashedGaussianMLPActor(nn.Module):

#     def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
#         super().__init__()
#         self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
#         self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
#         self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
#         self.act_limit = act_limit

#     def forward(self, obs, deterministic=False, with_logprob=True):
#         net_out = self.net(obs)
#         mu = self.mu_layer(net_out)
#         log_std = self.log_std_layer(net_out)
#         log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
#         std = torch.exp(log_std)

#         # Pre-squash distribution and sample
#         pi_distribution = Normal(mu, std)
#         if deterministic:
#             # Only used for evaluating policy at test time.
#             pi_action = mu
#         else:
#             pi_action = pi_distribution.rsample()

#         if with_logprob:
#             # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
#             # NOTE: The correction formula is a little bit magic. To get an understanding 
#             # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
#             # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
#             # Try deriving it yourself as a (very difficult) exercise. :)
#             logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1, keepdim=True)
#             logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1, keepdim=True)
#             # logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
#             # logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
#         else:
#             logp_pi = None

#         pi_action = torch.tanh(pi_action)
#         pi_action = self.act_limit * pi_action

#         return pi_action, logp_pi


# class MLPQFunction(nn.Module):

#     def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
#         super().__init__()
#         self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

#     def forward(self, obs, act):
#         q = self.q(torch.cat([obs, act], dim=-1))
#         # return torch.squeeze(q, -1) # Critical to ensure q has right shape.
#         return q

# class MLPActorCritic(nn.Module):

#     def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
#                  activation=nn.ReLU):
#         super().__init__()

#         obs_dim = observation_space.shape[0]
#         act_dim = action_space.shape[0]
#         act_limit = action_space.high[0]

#         # build policy and value functions
#         self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
#         self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
#         self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

#     def act(self, obs, deterministic=False):
#         with torch.no_grad():
#             a, _ = self.pi(obs, deterministic, False)
#             return a.cpu().numpy()

class Actor_SAC(nn.Module):
    def __init__(self, input_shape, action_space, hidden_dim=256, noisy_nets = False, noisy_sigma=0.5):
        super().__init__()

        self.noisy = noisy_nets
        num_outputs = action_space.shape[0]

        self.action_scale = torch.tensor((action_space.high - action_space.low) / 2., dtype=torch.float)
        self.action_bias = torch.tensor((action_space.high + action_space.low) / 2., dtype=torch.float)

        self.fc1 = nn.Linear(input_shape[0], hidden_dim) if not self.noisy else NoisyLinear(input_shape[0], hidden_dim, noisy_sigma)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) if not self.noisy else NoisyLinear(hidden_dim, hidden_dim, noisy_sigma)

        self.actor_mean = nn.Linear(hidden_dim, num_outputs) if not self.noisy else NoisyLinear(hidden_dim, num_outputs, noisy_sigma)
        self.actor_log_std = nn.Linear(hidden_dim, num_outputs) if not self.noisy else NoisyLinear(hidden_dim, num_outputs, noisy_sigma)

        if self.noisy:
            self.sample_noise()

    def forward(self, obs, deterministic=False, with_logprob=True):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))

        mean = self.actor_mean(x)
        log_std = self.actor_log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mean, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            action = mean
        else:
            action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # See appendix C of original paper
            logprob = pi_distribution.log_prob(action).sum(axis=-1, keepdim=True)
            logprob -= (2*(np.log(2) - action - F.softplus(-2*action))).sum(axis=1, keepdim=True)
        else:
            logprob = None

        action = torch.tanh(action)
        action = self.action_scale * action + self.action_bias

        return action, logprob

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.forward(obs, deterministic, False)
            return a.cpu().numpy()

    def sample_noise(self):
        if self.noisy:
            self.fc1.sample_noise()
            self.fc2.sample_noise()
            self.actor_mean.sample_noise()
            self.actor_log_std.sample_noise()

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        super().to(device)



class DQN_SAC(nn.Module):
    def __init__(self, input_shape, action_space, hidden_dim=256, noisy_nets=False, noisy_sigma=0.5):
        super().__init__()

        self.noisy = noisy_nets

        num_inputs = input_shape[0] + action_space.shape[0]

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs, hidden_dim) if not self.noisy else NoisyLinear(num_inputs, hidden_dim, noisy_sigma)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim) if not self.noisy else NoisyLinear(hidden_dim, hidden_dim, noisy_sigma)
        self.linear3 = nn.Linear(hidden_dim, 1) if not self.noisy else NoisyLinear(hidden_dim, 1, noisy_sigma)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs, hidden_dim) if not self.noisy else NoisyLinear(num_inputs, hidden_dim, noisy_sigma)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim) if not self.noisy else NoisyLinear(hidden_dim, hidden_dim, noisy_sigma)
        self.linear6 = nn.Linear(hidden_dim, 1) if not self.noisy else NoisyLinear(hidden_dim, 1, noisy_sigma)

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=-1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

    def sample_noise(self):
        if self.noisy:
            self.linear1.sample_noise()
            self.linear2.sample_noise()
            self.linear3.sample_noise()
            self.linear4.sample_noise()
            self.linear5.sample_noise()
            self.linear6.sample_noise()
