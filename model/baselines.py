# Define pytorch models here

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.input = input_dim
        self.hidden = hidden_dim
        self.output = out_dim
        self.fc1 = nn.Linear(self.input, self.hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden, self.output)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class SimpleVAE(nn.Module):
    """
    Simple VAE
    """
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(SimpleVAE, self).__init__()
        self.input = input_dim
        self.hidden = hidden_dim
        self.output = out_dim
        self.fc1 = nn.Linear(self.input, self.hidden)
        self.relu1 = nn.ReLU()
        self.fc21 = nn.Linear(self.hidden, self.output)
        self.fc22 = nn.Linear(self.hidden, self.output)
        self.fc3 = nn.Linear(self.output, 20)
        self.relu2 = nn.ReLU()
        self.fc4 = nn.Linear(20, 7)

    def encode(self, x):
        out = self.relu1(self.fc1(x))
        return self.fc21(out), self.fc22(out)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        out = self.relu2(self.fc3(z))
        return self.fc4(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

"""
Comes from https://git.uwaterloo.ca/pjaini/SOS-Flows
Implementation of SOS Flow
"""



def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output

    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('mask', mask)

    def forward(self, inputs):
        return F.linear(inputs, self.weight * self.mask, self.bias)


class ConditionerNet(nn.Module):
    def __init__(self, input_size, hidden_size, k, m, n_layers=1):
        super().__init__()
        self.k = k
        self.m = m
        self.input_size = input_size
        self.output_size = k * self.m * input_size + input_size
        self.network = self._make_net(input_size, hidden_size, self.output_size, n_layers)

    def _make_net(self, input_size, hidden_size, output_size, n_layers):
        if self.input_size > 1:
            input_mask = get_mask(
                input_size, hidden_size, input_size, mask_type='input')
            hidden_mask = get_mask(hidden_size, hidden_size, input_size)
            output_mask = get_mask(
                hidden_size, output_size, input_size, mask_type='output')

            network = nn.Sequential(
                MaskedLinear(input_size, hidden_size, input_mask), nn.ReLU(),
                MaskedLinear(hidden_size, hidden_size, hidden_mask), nn.ReLU(),
                MaskedLinear(hidden_size, output_size, output_mask))
        else:
            network = nn.Sequential(
                nn.Linear(input_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, output_size))

        '''
        for module in network.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                module.bias.data.fill_(0)
        '''

        return network

    def forward(self, inputs):
        batch_size = inputs.size(0)
        params = self.network(inputs)
        i = self.k * self.m * self.input_size
        c = params[:, :i].view(batch_size, -1, self.input_size).transpose(1,2).view(
            batch_size, self.input_size, self.k, self.m, 1)
        const = params[:, i:].view(batch_size, self.input_size)
        C = torch.matmul(c, c.transpose(3,4))
        return C, const


