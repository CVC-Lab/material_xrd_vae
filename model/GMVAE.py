import torch
import torch.nn as nn
import torch.nn.functional as F

class GMVAE(nn.Module):
    """
    Gaussian Mixture VAE
    """