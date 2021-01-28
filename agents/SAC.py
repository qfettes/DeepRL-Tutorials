import sys
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from networks.networks import DQN_SAC, Actor
from utils import ExponentialSchedule, LinearSchedule, PiecewiseSchedule
from utils.ReplayMemory import ExperienceReplayMemory, PrioritizedReplayMemory

from agents.BaseAgent import BaseAgent

np.set_printoptions(threshold=sys.maxsize)


class Agent(BaseAgent):
    def __init__(self, env=None, config=None, log_dir='/tmp/gym', tb_writer=None):
        super(Agent, self).__init__(env=env, config=config,
                                    log_dir=log_dir, tb_writer=tb_writer)

        assert(not config.double_dqn), "Double DQN is not supported with SAC"
        assert(
            not config.policy_gradient_recurrent_policy), "Recurrent Policy is not supported with SAC"
        assert((len(config.adaptive_repeat) == 1) and (config.adaptive_repeat[0] == 4)), f"Adaptive Repeat \
            isn't supported in continuous action spaces; it has been changed from its default value to \
                {config.adaptive_repeat}"
        assert(not config.dueling_dqn), "Dueling DQN is not supported with SAC"

        self.continousActionSpace = False
        if env.action_space.__class__.__name__ == 'Discrete':  # NOTE: shouldn't happen right now
            self.action_space = env.action_space.n * \
                len(config.adaptive_repeat)
        elif env.action_space.__class__.__name__ == 'Box':
            self.action_space = env.action_space
            self.continousActionSpace = True
        else:
            print('[ERROR] Unrecognized Action Space Type')
            exit()

        self.config = config
        self.num_feats = env.observation_space.shape
        self.envs = env

        self.declare_networks()
        optim_parameters = list(self.policy_net.parameters(
        )) + list(self.q_net_1.parameters()) + list(self.q_net_2.parameters())

        # See: https://arxiv.org/pdf/1812.05905.pdf
        if self.config.entropy_tuning is True:
            if self.continousActionSpace:
                self.target_entropy_coef = - \
                    torch.prod(torch.Tensor(env.action_space.shape).to(
                        self.config.device)).item()
            else:
                # See: https://arxiv.org/pdf/1910.07207.pdf
                self.target_entropy_coef = 0.98 * \
                    np.log(1. / env.action_space.n)

            self.log_entropy_coef = torch.zeros(
                1, requires_grad=True, device=self.config.device)
            optim_parameters += [self.log_entropy_coef]

        self.optimizer = optim.Adam(
            optim_parameters, lr=self.config.lr, eps=self.config.adam_eps)

        # self.value_loss_fun = torch.nn.SmoothL1Loss(reduction='none')
        self.value_loss_fun = torch.nn.MSELoss(reduction='mean')

        # move to correct device
        self.policy_net.to(self.config.device)
        self.q_net_1.to(self.config.device)
        self.q_net_2.to(self.config.device)
        self.target_q_net_1.to(self.config.device)
        self.target_q_net_2.to(self.config.device)

        if self.config.inference:
            self.policy_net.eval()
            self.q_net_1.eval()
            self.q_net_2.eval()
            self.target_q_net_1.eval()
            self.target_q_net_2.eval()
        else:
            self.policy_net.train()
            self.q_net_1.train()
            self.q_net_2.train()
            self.target_q_net_1.train()
            self.target_q_net_2.train()

        self.declare_memory()
        self.update_count = 0
        self.nstep_buffer = []

        self.first_action = True

        self.training_priors()

    def declare_networks(self):
        self.policy_net = Actor(self.num_feats, self.action_space, self.config.body_out,
                                self.config.policy_gradient_recurrent_policy, self.config.gru_size, self.config.noisy_nets, self.config.sigma_init)

        self.q_net_1 = DQN_SAC(self.num_feats, self.action_space,
                               noisy=self.config.noisy_nets, sigma_init=self.config.sigma_init)
        self.q_net_2 = DQN_SAC(self.num_feats, self.action_space,
                               noisy=self.config.noisy_nets, sigma_init=self.config.sigma_init)
        self.target_q_net_1 = DQN_SAC(self.num_feats, self.action_space,
                                      noisy=self.config.noisy_nets, sigma_init=self.config.sigma_init)
        self.target_q_net_2 = DQN_SAC(self.num_feats, self.action_space,
                                      noisy=self.config.noisy_nets, sigma_init=self.config.sigma_init)

        self.target_q_net_1.load_state_dict(self.q_net_1.state_dict())
        self.target_q_net_2.load_state_dict(self.q_net_2.state_dict())

    def declare_memory(self):
        if self.config.priority_replay:
            self.memory = PrioritizedReplayMemory(
                self.config.exp_replay_size, self.config.priority_alpha, self.config.priority_beta_start, self.config.priority_beta_tsteps)
        else:
            self.memory = ExperienceReplayMemory(self.config.exp_replay_size)

    def training_priors(self):
        self.episode_rewards = np.zeros(self.config.num_envs)
        self.last_100_rewards = deque(maxlen=100)

        if len(self.config.epsilon_final) == 1:
            if self.config.epsilon_decay[0] > 1.0:
                self.anneal_eps = ExponentialSchedule(
                    self.config.epsilon_start, self.config.epsilon_final[0], self.config.epsilon_decay[0], self.config.max_tsteps)
            else:
                self.anneal_eps = LinearSchedule(
                    self.config.epsilon_start, self.config.epsilon_final[0], self.config.epsilon_decay[0], self.config.max_tsteps)
        else:
            self.anneal_eps = PiecewiseSchedule(
                self.config.epsilon_start, self.config.epsilon_final, self.config.epsilon_decay, self.config.max_tsteps)

        self.prev_observations, self.actions, self.rewards, self.dones = None, None, None, None,
        self.observations = self.envs.reset()

    def append_to_replay(self, s, a, r, s_, t):
        # TODO: Naive. This is implemented like rainbow; however, true nstep
        # q learning requires off-policy correction
        self.nstep_buffer.append([s, a, r, s_, t])

        if(len(self.nstep_buffer) < self.config.N_steps):
            return

        R = np.zeros_like(r)
        T = np.zeros_like(t)
        for idx, transition in enumerate(reversed(self.nstep_buffer)):
            exp_ = len(self.nstep_buffer) - idx - 1
            R *= (1.0 - transition[4])
            R += (self.config.gamma**exp_) * transition[2]

            T += transition[4]

        S, A, _, _, _ = self.nstep_buffer.pop(0)

        for state, action, reward, next_state, terminal in zip(S, A, R, s_, T):
            next_state = None if terminal >= 1 else next_state
            self.memory.push((state, action, reward, next_state))

    def prep_minibatch(self, tstep):
        # random transition batch is taken from experience replay memory
        data, indices, weights = self.memory.sample(
            self.config.batch_size, tstep)
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values = data

        batch_state = torch.from_numpy(batch_state).to(
            self.config.device).to(torch.float)
        batch_action = torch.from_numpy(batch_action).to(
            self.config.device).to(torch.long).unsqueeze(dim=1)
        batch_reward = torch.from_numpy(batch_reward).to(
            self.config.device).to(torch.float).unsqueeze(dim=1)

        non_final_mask = torch.from_numpy(non_final_mask).to(
            self.config.device).to(torch.bool)
        if not empty_next_state_values:
            non_final_next_states = torch.from_numpy(
                non_final_next_states).to(self.config.device).to(torch.float)

        if self.config.priority_replay:
            weights = torch.from_numpy(weights).to(
                self.config.device).to(torch.float).view(-1, 1)

        batch_state /= 255.0
        non_final_next_states /= 255.0

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights

    def compute_loss(self, batch_vars, tstep):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        # TODO: fix this to work with both continuous actions, too
        # estimate
        self.q_net_1.sample_noise()
        self.q_net_2.sample_noise()
        current_q_values_1 = self.q_net_1(batch_state).gather(1, batch_action)
        current_q_values_2 = self.q_net_2(batch_state).gather(1, batch_action)

        # target
        with torch.no_grad():
            next_actions, next_action_log_probs, _ = self.get_action(
                non_final_next_states, states=None, masks=None, deterministic=False)

            # TODO: We can't use self.config.batch_size here in microservices
            #   Value function batches will be variable size. Remove this and
            #   all other instances of its use
            next_q_values_1 = torch.zeros(
                self.config.batch_size, device=self.config.device, dtype=torch.float).unsqueeze(dim=1)
            next_q_values_2 = torch.zeros(
                self.config.batch_size, device=self.config.device, dtype=torch.float).unsqueeze(dim=1)

            self.target_q_net_1.sample_noise()
            self.target_q_net_2.sample_noise()

            if not empty_next_state_values:
                next_q_values_1[non_final_mask] = (self.config.gamma**self.config.N_steps) * \
                    (self.target_q_net_1(non_final_next_states).gather(
                        1, next_actions) - (self.config.entropy_coef * next_action_log_probs))
                next_q_values_2[non_final_mask] = (self.config.gamma**self.config.N_steps) * \
                    (self.target_q_net_2(non_final_next_states).gather(
                        1, next_actions) - (self.config.entropy_coef * next_action_log_probs))

                next_q_values = torch.min(
                    torch.cat((next_q_values_1, next_q_values_2), dim=1),
                    dim=1,
                    keepdim=True)[0]

            target = batch_reward + next_q_values

        value_loss = self.value_loss_fun(current_q_values_1, target)
        value_loss += self.value_loss_fun(current_q_values_2, target)

        if self.config.priority_replay:
            with torch.no_grad():
                diff = torch.abs(2. * target - current_q_values_1 -
                                 current_q_values_2).squeeze().cpu().numpy().tolist()
                self.memory.update_priorities(indices, diff)
            value_loss *= weights

        value_loss = value_loss.mean()

        # Compute policy loss
        on_policy_actions, on_policy_action_log_probs, _ = self.get_action(
            batch_state, states=None, masks=None, deterministic=False)

        # with torch.no_grad():
        on_policy_q_values_1 = self.target_q_net_1(batch_state)
        on_policy_q_values_1 = on_policy_q_values_1.gather(
            1, on_policy_actions)

        on_policy_q_values_2 = self.target_q_net_2(batch_state)
        on_policy_q_values_2 = on_policy_q_values_2.gather(
            1, on_policy_actions)

        on_policy_q_values = torch.min(
            torch.cat((on_policy_q_values_1, on_policy_q_values_2), dim=1),
            dim=1, keepdim=True)[0]

        policy_loss = on_policy_q_values - \
            self.config.entropy_coef * on_policy_action_log_probs
        policy_loss = -1. * policy_loss.mean()

        loss = value_loss + policy_loss

        if self.config.entropy_tuning:
            entropy_coef_loss = -(self.log_entropy_coef * (
                on_policy_action_log_probs + self.target_entropy_coef).detach()).mean()
            loss += entropy_coef_loss

            self.config.entropy_coef = self.log_entropy_coef.exp().item()

            with torch.no_grad():
                self.tb_writer.add_scalar(
                    'Loss/Entropy Coefficient Loss', entropy_coef_loss.item(), tstep)

        # log val estimates
        with torch.no_grad():
            self.tb_writer.add_scalar('Policy/Value Estimate', torch.cat(
                (current_q_values_1, current_q_values_2)).detach().mean().item(), tstep)
            self.tb_writer.add_scalar(
                'Policy/Next Value Estimate', target.detach().mean().item(), tstep)
            self.tb_writer.add_scalar(
                'Policy/Entropy Coefficient', self.config.entropy_coef, tstep)
            self.tb_writer.add_scalar(
                'Loss/Value Loss', value_loss.item(), tstep)
            self.tb_writer.add_scalar(
                'Loss/Policy Loss', policy_loss.item(), tstep)

        return loss

    def update_(self, tstep=0):
        # TODO: add support for more than one update here
        #   also add support for this to DQNu

        if tstep < self.config.learn_start:
            return None

        batch_vars = self.prep_minibatch(tstep)

        loss = self.compute_loss(batch_vars, tstep)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), self.config.grad_norm_max)
        torch.nn.utils.clip_grad_norm_(
            self.q_net_1.parameters(), self.config.grad_norm_max)
        torch.nn.utils.clip_grad_norm_(
            self.q_net_2.parameters(), self.config.grad_norm_max)
        self.optimizer.step()

        self.update_target_model()

        # more logging
        with torch.no_grad():
            self.tb_writer.add_scalar('Loss/Total Loss', loss.item(), tstep)
            self.tb_writer.add_scalar('Learning/Learning Rate', np.mean(
                [param_group['lr'] for param_group in self.optimizer.param_groups]), tstep)

            # log weight norm
            weight_norm = 0.
            for p in self.q_net_1.parameters():
                param_norm = p.data.norm(2)
                weight_norm += param_norm.item() ** 2
            weight_norm = weight_norm ** (1./2.)
            self.tb_writer.add_scalar(
                'Learning/Weight Norm', weight_norm, tstep)

            # log grad_norm
            grad_norm = 0.
            for p in self.q_net_1.parameters():
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm ** (1./2.)
            self.tb_writer.add_scalar('Learning/Grad Norm', grad_norm, tstep)

            # log sigma param norm
            if self.config.noisy_nets:
                sigma_norm = 0.
                for name, p in self.q_net_1.named_parameters():
                    if p.requires_grad and 'sigma' in name:
                        param_norm = p.data.norm(2)
                        sigma_norm += param_norm.item() ** 2
                sigma_norm = sigma_norm ** (1./2.)
                self.tb_writer.add_scalar(
                    'Policy/Sigma Norm', sigma_norm, tstep)

    def get_action(self, s, states, masks, deterministic=False):
        logits, states = self.policy_net(s, states, masks)

        # TODO: clean this up
        if self.continousActionSpace:
            dist = torch.distributions.Normal(
                logits, F.softplus(self.policy_net.logstd))
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

        return actions, action_log_probs, states

    def update_target_model(self):
        for target_param, param in zip(self.target_q_net_1.parameters(), self.q_net_1.parameters()):
            target_param.data.copy_(
                self.config.polyak_coef * target_param + (1. - self.config.polyak_coef) * param)

        for target_param, param in zip(self.target_q_net_2.parameters(), self.q_net_2.parameters()):
            target_param.data.copy_(
                self.config.polyak_coef * target_param + (1. - self.config.polyak_coef) * param)

    def add_graph(self, inp):
        with torch.no_grad():
            X = torch.from_numpy(inp).to(self.config.device).to(
                torch.float).view((-1,)+self.num_feats)
            self.tb_writer.add_graph(self.q_net_1, X)
            self.first_action = False

    def reset_hx(self, idx):
        pass

    def step(self, current_tstep, step=0):
        epsilon = self.anneal_eps(current_tstep)
        self.tb_writer.add_scalar('Policy/Epsilon', epsilon, current_tstep)

        # TODO: modifying step in the next line is incosistent with prior code style
        X = torch.from_numpy(self.observations).to(self.config.device).to(
            torch.float).view((-1,)+self.num_feats) / 255.0
        self.actions, _, _ = self.get_action(
            X, states=None, masks=None, deterministic=False)
        self.actions = self.actions.view(-1).cpu().numpy()

        self.prev_observations = self.observations
        self.observations, self.rewards, self.dones, self.infos = self.envs.step(
            self.actions)

        self.append_to_replay(self.prev_observations, self.actions,
                              self.rewards, self.observations, self.dones.astype(int))

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

    def update(self, current_tstep):
        self.update_(current_tstep)

    # TODO: Fix saving
    def save_w(self):
        pass

    # TODO: Fix loading
    def load_w(self):
        pass
