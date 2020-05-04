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
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(SimpleVAE, self).__init__()
        self.input = input_dim
        self.hidden = hidden_dim
        self.latent = latent_dim
        self.fc1 = nn.Linear(self.input, self.hidden)
        self.relu1 = nn.ReLU()
        self.fc21 = nn.Linear(self.hidden, self.latent)
        self.fc22 = nn.Linear(self.hidden, self.latent)
        self.fc3 = nn.Linear(self.latent, self.hidden)
        self.relu2 = nn.LeakyReLU()
        self.fc4 = nn.Linear(self.hidden, self.input)

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

