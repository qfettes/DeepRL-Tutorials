import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.layers import NoisyLinear


class AtariBody(nn.Module):
    def __init__(self, input_shape, num_actions, noisy_nets=False, noisy_sigma=0.5):
        super(AtariBody, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.noisy = noisy_nets

        self.conv1 = nn.Conv2d(
            self.input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # NOTE: got initialization tip from: https://towardsdatascience.com/tutorial-double-deep-q-learning-with-dueling-network-architectures-4c1b3fb7f756
        #   Is this correct?
        # torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        # torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        # torch.nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        return x

    def feature_size(self):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape)))).view(1, -1).size(1)

    def sample_noise(self):
        pass


class SimpleBody(nn.Module):
    def __init__(self, input_shape, num_actions, noisy_nets=False, noisy_sigma=0.5):
        super(SimpleBody, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.noisy = noisy_nets

        self.fc1 = nn.Linear(input_shape[0], 128) if not self.noisy else NoisyLinear(
            input_shape[0], 128, noisy_sigma)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x

    def feature_size(self):
        return self.fc1(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

    def sample_noise(self):
        if self.noisy:
            self.fc1.sample_noise()


class AtariBodyAC(nn.Module):
    def __init__(self, input_shape, body_out=512, noisy_nets=False, noisy_sigma=0.5):
        super(AtariBodyAC, self).__init__()
        self.body_out = body_out
        self.noisy_nets = noisy_nets
        self.noisy_sigma = noisy_sigma

        def init_(m): return self.layer_init(m, nn.init.orthogonal_,
                                             lambda x: nn.init.constant_(x, 0),
                                             nn.init.calculate_gain('relu'))

        self.conv1 = init_(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4))
        self.conv2 = init_(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = init_(nn.Conv2d(64, 64, kernel_size=3, stride=1))

        def init_(m): return self.layer_init(m, nn.init.orthogonal_,
                                             lambda x: nn.init.constant_(x, 0),
                                             nn.init.calculate_gain('relu'),
                                             noisy_layer=self.noisy_nets)

        self.fc1 = init_(nn.Linear(self.feature_size(input_shape), body_out)) if not self.noisy_nets else init_(
            NoisyLinear(self.feature_size(input_shape), body_out, noisy_sigma))

        self.train()
        if self.noisy_nets:
            self.sample_noise()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        return x

    def layer_init(self, module, weight_init, bias_init, gain=1, noisy_layer=False):
        if not noisy_layer:
            weight_init(module.weight.data, gain=gain)
            bias_init(module.bias.data)
        else:
            weight_init(module.weight_mu.data, gain=gain)
            bias_init(module.bias_mu.data)
        return module

    def feature_size(self, input_shape):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *input_shape)))).view(1, -1).size(1)

    def sample_noise(self):
        if self.noisy:
            self.fc1.sample_noise()


class SimpleBodyAC(nn.Module):
    def __init__(self, input_shape, output_shape=200, noisy_nets=False, noisy_sigma=0.5):
        super(SimpleBodyAC, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.noisy_nets = noisy_nets

        def init_(m): return self.layer_init(m, nn.init.orthogonal_,
                                             lambda x: nn.init.constant_(x, 0),
                                             np.sqrt(2),
                                             noisy_layer=self.noisy_nets)

        self.fc1 = init_(nn.Linear(input_shape[0], output_shape)) if not self.noisy_nets else init_(
            NoisyLinear(input_shape[0], output_shape, noisy_sigma))

        self.train()
        if self.noisy_nets:
            self.sample_noise()

    def forward(self, inp):
        x = torch.relu(self.fc1(inp))

        return x

    def layer_init(self, module, weight_init, bias_init, gain=1, noisy_layer=False):
        if not noisy_layer:
            weight_init(module.weight.data, gain=gain)
            bias_init(module.bias.data)
        else:
            weight_init(module.weight_mu.data, gain=gain)
            bias_init(module.bias_mu.data)
        return module

    def feature_size(self, input_shape):
        return self.fc1(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

    def sample_noise(self):
        if self.noisy_nets:
            self.fc1.sample_noise()
