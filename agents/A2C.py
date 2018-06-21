import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from agents.BaseAgent import BaseAgent
from networks.networks import ActorCritic

from timeit import default_timer as timer

class Model(BaseAgent):
    def __init__(self, static_policy=False, env=None, config=None):
        super(Model, self).__init__()
        self.device = config.device

        self.noisy=config.USE_NOISY_NETS
        self.priority_replay=config.USE_PRIORITY_REPLAY

        self.gamma = config.GAMMA
        self.lr = config.LR
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.learn_start = config.LEARN_START
        self.sigma_init= config.SIGMA_INIT
        self.num_agents = config.num_agents
        self.value_loss_weight = config.value_loss_weight
        self.entropy_loss_weight = config.entropy_loss_weight

        self.static_policy = static_policy
        self.num_feats = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.env = env

        self.declare_networks()
            
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        #move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)

        if self.static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()

        self.update_count = 0

        self.nsteps = config.N_STEPS
        self.nstep_buffer = []

    def declare_networks(self):
        self.model = ActorCritic(self.num_feats, self.num_actions, noisy=self.noisy, sigma_init=self.sigma_init)
        self.target_model = ActorCritic(self.num_feats, self.num_actions, noisy=self.noisy, sigma_init=self.sigma_init)

    def get_action(self, s, eps=0.1):
        X = torch.tensor(s, device=self.device, dtype=torch.float)
        self.model.sample_noise()
        policy, value = self.model(X)
        return policy, value

    def process_rollout(self, mem):
        _, _, _, _, last_values = mem[-1]
        returns = last_values

        advantages = torch.zeros(self.num_agents, 1, device=self.device, dtype=torch.float)

        out = [None] * (len(mem) - 1)

        # run Generalized Advantage Estimation, calculate returns, advantages
        for t in reversed(range(len(mem) - 1)):
            rewards, actions, masks, policies, values = mem[t]
            _, _, _, _, next_values = mem[t + 1]

            with torch.no_grad():
                rewards = torch.tensor(rewards, device=self.device, dtype=torch.float).view(-1, 1)
                actions = torch.tensor(actions, device=self.device, dtype=torch.long).view(-1, 1)
                masks = torch.tensor(masks, device=self.device, dtype=torch.float).view(-1, 1)

                returns = rewards + returns * self.gamma * masks

                deltas = rewards + next_values * self.gamma * masks - values
                advantages = advantages * self.gamma * 1.00 * masks + deltas

            out[t] = actions, policies, values, returns, advantages

            
        a, p, v, r, adv = zip(*out)
        a = torch.cat(a, 0)
        p = torch.cat(p, 0)
        v = torch.cat(v, 0)
        r = torch.cat(r, 0)
        adv = torch.cat(adv, 0)

        return a, p, v, r, adv

    def compute_loss(self, rollout):
        actions, policies, values, returns, advantages = rollout

        probs = F.softmax(policies, dim=1)
        log_probs = F.log_softmax(policies, dim=1)
        log_action_probs = log_probs.gather(1, actions)

        policy_loss = (-log_action_probs * advantages.detach()).mean()
        value_loss = (.5 * (values - returns) ** 2.).mean()
        entropy_loss = (log_probs * probs).sum(dim=1).mean()

        loss = policy_loss + value_loss * self.value_loss_weight + entropy_loss * self.entropy_loss_weight

        return loss

    def update(self, mem):
        # bootstrap discounted returns with final value estimates
        rollout = self.process_rollout(mem)

        loss = self.compute_loss(rollout)

        #nn.utils.clip_grad_norm(net.parameters(), args.grad_norm_limit)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        #self.update_target_model()
        self.save_loss(loss.item())
        self.save_sigma_param_magnitudes()