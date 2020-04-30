import torch
import numpy as np
from torch import nn, optim
from model.SOSFlowNet import SOSFlowNet,BatchNormFlow,Reverse,FlowSequential
from loss_function import GMVAELossFunctions
from metric import Metrics
import matplotlib.pyplot as plt

def build_model(input_size, hidden_size, k, r, n_blocks, device=None, **kwargs):
    modules = []
    for i in range(n_blocks):
        modules += [
            SOSFlowNet(input_size, hidden_size, k, r),
            BatchNormFlow(input_size),
            Reverse(input_size)
        ]
    model = FlowSequential(*modules)
    if device is not None:
        model.to(device)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
   
    return model


class SOSFlow:
    def __init__(self, args):
        self.input_size = args.data_size
        self.hidden_size = args.hidden_size
        self.k = args.k

        self.device = torch.device("cuda:%d" % args.gpu if args.cuda else "cpu")
        self.network = build_model(self.input_size, self.hidden_size, self.k, self.r, self.n_blocks, self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)


    def flow_loss(z, logdet, size_average=True, use_cuda=True):
        # If using Student-t as source distribution#
        #df = torch.tensor(5.0)
        #if use_cuda:
        #   log_prob = log_prob_st(z, torch.tensor([5.0]).cuda())
        #else:
            #log_prob = log_prob_st(z, torch.tensor([5.0]))
        #log_probs = log_prob.sum(-1, keepdim=True)
        ''' If using Uniform as source distribution
        log_probs = 0
        '''
        log_probs = (-0.5 * z.pow(2) - 0.5 * np.log(2 * np.pi)).sum(-1, keepdim=True)
        loss = -(log_probs + logdet).sum()
        # CHANGED TO UNIFORM SOURCE DISTRIBUTION
        #loss = -(logdet).sum()
        if size_average:
            loss /= z.size(0)
        return loss
