import numpy as np
import torch
import torch.optim as optim
from networks.network_bodies import AtariBody, SimpleBody
from networks.networks import DRQN
from utils.hyperparameters import Config
from utils.ReplayMemory import RecurrentExperienceReplayMemory

from agents.DQN import Agent as DQN_Agent


class Agent(DQN_Agent):
    def __init__(self, env=None, config=None, log_dir='/tmp/gym'):
        self.sequence_length = config.drqn_sequence_length

        super().__init__(env=env, config=config,
                                    log_dir=log_dir, tb_writer=tb_writer)

        self.reset_hx()

    def declare_networks(self):
        self.q_net = DRQN(self.num_feats, self.num_actions, body=SimpleBody)
        self.target_q_net = DRQN(
            self.num_feats, self.num_actions, body=SimpleBody)

    def declare_memory(self):
        self.memory = RecurrentExperienceReplayMemory(
            self.experience_replay_size, self.sequence_length)
        #self.memory = ExperienceReplayMemory(self.experience_replay_size)

    def prep_minibatch(self):
        transitions, indices, weights = self.memory.sample(self.batch_size)

        batch_state, batch_action, batch_reward, batch_next_state = zip(
            *transitions)

        shape = (self.batch_size, self.sequence_length)+self.num_feats

        batch_state = torch.tensor(
            batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).view(
            self.batch_size, self.sequence_length, -1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).view(
            self.batch_size, self.sequence_length)
        # get set of next states for end of each sequence
        batch_next_state = tuple([batch_next_state[i] for i in range(
            len(batch_next_state)) if (i+1) % (self.sequence_length) == 0])

        non_final_mask = torch.tensor(tuple(map(
            lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.uint8)
        try:  # sometimes all next states are false, especially with nstep returns
            non_final_next_states = torch.tensor(
                [s for s in batch_next_state if s is not None], device=self.device, dtype=torch.float).unsqueeze(dim=1)
            non_final_next_states = torch.cat(
                [batch_state[non_final_mask, 1:, :], non_final_next_states], dim=1)
            empty_next_state_values = False
        except:
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights

    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        # estimate
        current_q_values, _ = self.q_net(batch_state)
        current_q_values = current_q_values.gather(2, batch_action).squeeze()

        # target
        with torch.no_grad():
            max_next_q_values = torch.zeros(
                (self.batch_size, self.sequence_length), device=self.device, dtype=torch.float)
            if not empty_next_state_values:
                max_next, _ = self.target_q_net(non_final_next_states)
                max_next_q_values[non_final_mask] = max_next.max(dim=2)[0]
            expected_q_values = batch_reward + \
                ((self.gamma**self.nsteps)*max_next_q_values)

        diff = (expected_q_values - current_q_values)
        loss = self.huber(diff)
        loss = loss.mean()

        return loss

    def get_action(self, s, eps=0.1):
        with torch.no_grad():
            self.seq.pop(0)
            self.seq.append(s)
            if np.random.random() >= eps or self.noisy:
                X = torch.tensor(
                    [self.seq], device=self.device, dtype=torch.float)
                self.q_net.sample_noise()
                a, _ = self.q_net(X)
                a = a[:, -1, :]  # select last element of seq
                a = a.max(1)[1]
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)

    # def get_max_next_state_action(self, next_states, hx):
    #    max_next, _ = self.target_q_net(next_states, hx)
    #    return max_next.max(dim=1)[1].view(-1, 1)'''

    def reset_hx(self):
        #self.action_hx = self.q_net.init_hidden(1)
        self.seq = [np.zeros(self.num_feats)
                    for j in range(self.sequence_length)]
