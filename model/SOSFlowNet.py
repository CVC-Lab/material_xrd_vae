#
#   SOS Block:
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class SOSFlow(nn.Module):
    @staticmethod
    def power(z, k):
        return z ** (torch.arange(k).float().to(z.device))

    def __init__(self, input_size, hidden_size, k, r, n_layers=1):
        super().__init__()
        self.k = k
        self.m = r+1

        self.conditioner = ConditionerNet(input_size, hidden_size, k, self.m, n_layers)
        self.register_buffer('filter', self._make_filter())

    def _make_filter(self):
        n = torch.arange(self.m).unsqueeze(1)
        e = torch.ones(self.m).unsqueeze(1).long()
        filter = (n.mm(e.transpose(0, 1))) + (e.mm(n.transpose(0, 1))) + 1
        return filter.float()

    def forward(self, inputs, mode='direct'):
        batch_size, input_size = inputs.size(0), inputs.size(1)
        C, const = self.conditioner(inputs)
        X = SOSFlow.power(inputs.unsqueeze(-1), self.m).view(batch_size, input_size, 1, self.m,
                                                              1)  # bs x d x 1 x m x 1
        Z = self._transform(X, C / self.filter) * inputs + const
        logdet = torch.log(torch.abs(self._transform(X, C))).sum(dim=1, keepdim=True)
        return Z, logdet

    def _transform(self, X, C):
        CX = torch.matmul(C, X)                                                                 # bs x d x k x m x 1
        XCX = torch.matmul(X.transpose(3, 4), CX)                                               # bs x d x k x 1 x 1
        summed = XCX.squeeze(-1).squeeze(-1).sum(-1)                                            # bs x d
        return summed

    def _jacob(self, inputs, mode='direct'):
        X = inputs.clone()
        X.requires_grad_()
        X.retain_grad()
        d = X.size(0)

        X_in = X.unsqueeze(0)
        C, const = self.conditioner(X_in)
        Xpow = SOSFlow.power(X_in.unsqueeze(-1), self.m).view(1, d, 1, self.m,
                                                             1)  # bs x d x 1 x m x 1
        Z = (self._transform(Xpow, C / self.filter) * X_in + const).view(-1)

        J = torch.zeros(d,d)
        for i in range(d):
            self.zero_grad()
            Z[i].backward(retain_graph=True)
            J[i,:] = X.grad

        del X, X_in, C, const, Xpow, Z
        return J



