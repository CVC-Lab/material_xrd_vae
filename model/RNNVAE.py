# Using RNN-type network as encoder/decoder

import torch
import torch.nn as nn

class GRUNet(nn.Module):
def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2, is_cuda=True):
    super(GRUNet, self).__init__()
    self.hidden_dim = hidden_dim
    self.n_layers = n_layers
    self.is_cuda = is_cuda
    self.fc0 = nn.Linear(input_dim, hidden_dim)
    self.relu = nn.ReLU()
    self.gru = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
    self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out = self.fc0(x)
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).requires_grad_()
        if self.is_cuda:
            h0 = h0.cuda()
        out, hn = self.gru(out, h0.detach())
        out = self.fc(out[-1,:, :])
        return out
