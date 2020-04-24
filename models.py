# Define pytorch models here

from torch import nn
from torch.nn import functional as F
import torch



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
