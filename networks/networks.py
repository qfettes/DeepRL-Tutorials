import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.layers import NoisyLinear

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions, noisy=False, sigma_init=0.5):
        super(DQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.noisy=noisy

        self.conv1 = nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(self.feature_size(), 512) if not self.noisy else NoisyLinear(self.feature_size(), 512, sigma_init)
        self.fc2 = nn.Linear(512, self.num_actions) if not self.noisy else NoisyLinear(512, self.num_actions, sigma_init)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    def feature_size(self):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape)))).view(1, -1).size(1)

    def sample_noise(self):
        if self.noisy:
            self.fc1.sample_noise()
            self.fc2.sample_noise()


class DQN_simple(nn.Module):
    def __init__(self, input_shape, num_actions, noisy=False, sigma_init=0.5):
        super(DQN_simple, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.noisy=noisy

        self.fc1 = nn.Linear(input_shape[0], 128) if not self.noisy else NoisyLinear(input_shape[0], 128, sigma_init)
        self.fc2 = nn.Linear(128, 128) if not self.noisy else NoisyLinear(128, 128, sigma_init)
        self.fc3 = nn.Linear(128, self.num_actions) if not self.noisy else NoisyLinear(128, self.num_actions, sigma_init)  
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_noise(self):
        if self.noisy:
            self.fc1.sample_noise()
            self.fc2.sample_noise()
            self.fc3.sample_noise()


class DuelingDQN(nn.Module):
    def __init__(self, input_shape, num_outputs, noisy=False, sigma_init=0.5):
        super(DuelingDQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_outputs
        self.noisy=noisy
        
        self.conv1 = nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.adv1 = nn.Linear(self.feature_size(), 512) if not self.noisy else NoisyLinear(self.feature_size(), 512, sigma_init)
        self.adv2 = nn.Linear(512, self.num_actions) if not self.noisy else NoisyLinear(512, self.num_actions, sigma_init)

        self.val1 = nn.Linear(self.feature_size(), 512) if not self.noisy else NoisyLinear(self.feature_size(), 512, sigma_init)
        self.val2 = nn.Linear(512, 1) if not self.noisy else NoisyLinear(512, 1, sigma_init)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv)

        val = F.relu(self.val1(x))
        val = self.val2(val)

        return val + adv - adv.mean()
    
    def feature_size(self):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape)))).view(1, -1).size(1)

    def sample_noise(self):
        if self.noisy:
            self.adv1.sample_noise()
            self.adv2.sample_noise()
            self.val1.sample_noise()
            self.val2.sample_noise()


class DuelingDQN_simple(nn.Module):
    def __init__(self, input_shape, num_outputs, noisy=False, sigma_init=0.5):
        super(DuelingDQN_simple, self).__init__()
        self.input_shape = input_shape
        self.num_outputs = num_outputs
        self.noisy=noisy
        
        self.fc1 = nn.Linear(self.input_shape[0], 128) if not self.noisy else NoisyLinear(self.input_shape[0], 128, sigma_init)

        self.adv1 = nn.Linear(128, 128) if not self.noisy else NoisyLinear(128, 128, sigma_init)
        self.adv2 = nn.Linear(128, self.num_outputs) if not self.noisy else NoisyLinear(128, self.num_outputs, sigma_init)

        self.val1 = nn.Linear(128, 128) if not self.noisy else NoisyLinear(128, 128, sigma_init)
        self.val2 = nn.Linear(128, 1) if not self.noisy else NoisyLinear(128, 1, sigma_init)
        
    def forward(self, x):
        x = self.fc1(x)

        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv)

        val = F.relu(self.val1(x))
        val = self.val2(val)

        return val + adv - adv.mean()

    def sample_noise(self):
        if self.noisy:
            self.fc1.sample_noise()
            self.adv1.sample_noise()
            self.adv2.sample_noise()
            self.val1.sample_noise()
            self.val2.sample_noise()