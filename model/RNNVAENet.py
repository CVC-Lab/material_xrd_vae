# Using RNN-type network as encoder/decoder

import torch
import random
import torch.nn as nn
from model.NormalizingFlowNet import PlanarTransformation, NormalizingFlowNet
from model.baselines import MLP_2layer

class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2, device=None):
        super(GRUEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.fc0 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(hidden_dim, output_dim, n_layers, batch_first=True, dropout=drop_prob)
        
    def forward(self, x):
        out = self.fc0(x)
        out = self.relu(out)
        #h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).requires_grad_()
        #if self.device is not None:
        #    h0 = h0.to(self.device)
        out, hn = self.gru(out) 
        #out dim is [seq_len, batch, num_directions * hidden_size]
        return out, hn

class GRUDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2, device=None):
        super(GRUDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.fc0 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(hidden_dim, output_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(output_dim, input_dim)

    def forward(self, x, hidden):
        out = self.relu(self.fc0(x))
        out = out.unsqueeze(1)
        #h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).requires_grad_()
        #if self.device is not None:
        #    h0 = h0.to(self.device)
        out, hn = self.gru(out, hidden) 
        out = self.fc(out.squeeze(1))
        return out, hn



class RNNVAENet(nn.Module):
    def __init__(self, input_dim, gru_hidden_dim, flow_hidden_dim, output_dim, predict_dim, n_flow_layers, device):
        super(RNNVAENet, self).__init__()
        self.device = device
        self.encoder = GRUEncoder(input_dim, gru_hidden_dim, gru_hidden_dim, 2, device=device)
        self.decoder = GRUDecoder(input_dim, gru_hidden_dim, gru_hidden_dim, 2, device=device)
        self.flow = NormalizingFlowNet(PlanarTransformation, flow_hidden_dim, n_flow_layers)
        self.fc_mean = nn.Linear(gru_hidden_dim, flow_hidden_dim)
        self.fc_var = nn.Linear(gru_hidden_dim, flow_hidden_dim)
        self.fc_flow = nn.Linear(gru_hidden_dim, self.flow.nParams * n_flow_layers)
        self.prediction_net = MLP_2layer(flow_hidden_dim, int(flow_hidden_dim/2), predict_dim)
        self.lift_map = MLP_2layer(flow_hidden_dim, gru_hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        """Use mean and variance to generate latent variables z
        
        Args
        mu      -- mean from encode()
        logvar  -- log variance from encode()

        Returns latent variables z
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    

    def forward(self, x, recon_x, teaching_ratio = 0.5):
        # Assume x in [batch_size, len, dim] format
        out, hidden_state = self.encoder(x)
        out = out[:,-1,:].squeeze(1)
        mu = self.fc_mean(out)
        logvar = self.fc_var(out)
        params = self.fc_flow(out).mean(dim=0).chunk(self.flow.K, dim=0)
        z = self.reparameterize(mu, logvar)
        z = self.flow.forward(z, params)

        trg_len = x.shape[1]
        batch_size = x.shape[0]
        trg_dim = x.shape[2]
        prediction = self.prediction_net(z)
        decoded = torch.zeros(batch_size, trg_len, trg_dim).to(self.device)

        input = self.lift_map(z)
        for t in range(0,trg_len):
            output, hidden_state= self.decoder(input, hidden_state)
            decoded[:,t,:] = output
            teacher_force = random.random() < teaching_ratio
            input = recon_x[:,t,:] if teacher_force else output

        return decoded, prediction
