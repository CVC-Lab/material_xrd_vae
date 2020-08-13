# Crazy model where 36 same networks are applied for 36 compressed XRD signals
from torch import nn
import torch
from .baselines import MLP_2layer

class split_VAE(nn.Module):
    def __init__(self, input_dim, input_modals, latent_dim):
        super(split_VAE, self).__init__()
        self.encoders = [MLP_2layer(input_dim, int((input_dim+latent_dim)/4), latent_dim) for i in range(input_modals)]
        self.decoders = [MLP_2layer(latent_dim, int((input_dim+latent_dim)/4), input_dim) for i in range(input_modals)]
        self.total_latent = latent_dim * input_modals
        self.predictor = MLP_2layer(self.total_latent, int((1+self.total_latent)/4), 1)
        


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out