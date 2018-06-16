import numpy as np

import torch

from agents.DQN import Model as DQN_Agent
from networks.network_bodies import RecurrentSimpleBody
from networks.networks import DQN
from utils.ReplayMemory import RecurrentExperienceReplayMemory
from utils.hyperparameters import SEQUENCE_LENGTH, device

class Model(DQN_Agent):
    def __init__(self, static_policy=False, env=None):
        self.sequence_length=SEQUENCE_LENGTH

        super(Model, self).__init__(static_policy, env)

        self.seq = [np.zeros(self.num_feats) for j in range(self.sequence_length)]
    
    
    def declare_networks(self):
        self.model = DQN(self.env.observation_space.shape, self.env.action_space.n, noisy=self.noisy, sigma_init=self.sigma_init, body=RecurrentSimpleBody)
        self.target_model = DQN(self.env.observation_space.shape, self.env.action_space.n, noisy=self.noisy, sigma_init=self.sigma_init, body=RecurrentSimpleBody)

    def declare_memory(self):
        self.memory = RecurrentExperienceReplayMemory(self.experience_replay_size, self.sequence_length)

    def prep_minibatch(self):
        transitions, indices, weights = self.memory.sample(self.batch_size)

        transitions = [trans for seq in transitions for trans in seq] #flatten to prepare
        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        batch_state = torch.cat(batch_state).view(self.batch_size, self.sequence_length, -1).transpose(0, 1)
        batch_action = torch.cat(batch_action).view(self.batch_size, self.sequence_length, -1).transpose(0, 1)[self.sequence_length-1]
        batch_reward = torch.cat(batch_reward).view(self.batch_size, self.sequence_length, -1).transpose(0, 1)[self.sequence_length-1]
        #get set of next states for end of each sequence
        batch_next_state = next_states = tuple([batch_next_state[i] for i in range(len(batch_next_state)) if (i+1)%(self.sequence_length)==0])

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=device, dtype=torch.uint8)
        try: #sometimes all next states are false, especially with nstep returns
            non_final_next_states = torch.cat([s for s in batch_next_state if s is not None])
            empty_next_state_values = False
        except:
            empty_next_state_values = True

        non_final_next_states = torch.cat((batch_state[1:, non_final_mask, :], non_final_next_states.unsqueeze(dim=0)))

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights

    def get_action(self, s, eps=0.1):
        with torch.no_grad():
            self.seq.pop(0)
            self.seq.append(s)
            if np.random.random() >= eps or self.static_policy or self.noisy:
                X = torch.tensor(self.seq, device=device, dtype=torch.float) 
                self.model.sample_noise()
                a = self.model(X).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)

    