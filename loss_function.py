import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from utils import safe_log


def simplevae_elbo_loss_function(recon_x, x, mu, logvar):
    MSE = nn.MSELoss(reduction='sum')(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD

def simplevae_elbo_loss_function_with_energy(recon_x, x, mu, logvar, pred_e, e):
    MSE = nn.MSELoss(reduction='mean')(recon_x, x)
    MSE_eng = nn.MSELoss(reduction='sum')(pred_e, e)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + MSE_eng + KLD, MSE, MSE_eng, KLD

class GMVAELossFunctions:
    eps = 1e-8

    def mean_squared_error(self, real, predictions):
        """Mean Squared Error between the true and predicted outputs
            loss = (1/n)*Σ(real - predicted)^2
        Args:
            real: (array) corresponding array containing the true labels
            predictions: (array) corresponding array containing the predicted labels

        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
        """
        loss = (real - predictions).pow(2)
        return loss.sum(-1).mean()


    def reconstruction_loss(self, real, predicted, rec_type='mse' ):
        """Reconstruction loss between the true and predicted outputs
            mse = (1/n)*Σ(real - predicted)^2
            bce = (1/n) * -Σ(real*log(predicted) + (1 - real)*log(1 - predicted))
        Args:
            real: (array) corresponding array containing the true labels
            predictions: (array) corresponding array containing the predicted labels

        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
        """
        if rec_type == 'mse':
            loss = (real - predicted).pow(2)
        elif rec_type == 'bce':
            loss = F.binary_cross_entropy(predicted, real, reduction='none')
        else:
            raise "invalid loss function... try bce or mse..."
        return loss.sum(-1).mean()


    def log_normal(self, x, mu, var):
        """Logarithm of normal distribution with mean=mu and variance=var
            log(x|μ, σ^2) = loss = -0.5 * Σ log(2π) + log(σ^2) + ((x - μ)/σ)^2
        Args:
            x: (array) corresponding array containing the input
            mu: (array) corresponding array containing the mean 
            var: (array) corresponding array containing the variance
        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
        """
        if self.eps > 0.0:
            var = var + self.eps
        return -0.5 * torch.sum(
        np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1)


    def gaussian_loss(self, z, z_mu, z_var, z_mu_prior, z_var_prior):
        """Variational loss when using labeled data without considering reconstruction loss
            loss = log q(z|x,y) - log p(z) - log p(y)
        Args:
            z: (array) array containing the gaussian latent variable
            z_mu: (array) array containing the mean of the inference model
            z_var: (array) array containing the variance of the inference model
            z_mu_prior: (array) array containing the prior mean of the generative model
            z_var_prior: (array) array containing the prior variance of the generative mode
            
        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
        """
        loss = self.log_normal(z, z_mu, z_var) - self.log_normal(z, z_mu_prior, z_var_prior)
        return loss.mean()


    def entropy(self, logits, targets):
        """Entropy loss
            loss = (1/n) * -Σ targets*log(predicted)
        Args:
            logits: (array) corresponding array containing the logits of the categorical variable
            real: (array) corresponding array containing the true labels

        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
        """
        log_q = F.log_softmax(logits, dim=-1)
        return -torch.mean(torch.sum(targets * log_q, dim=-1))


class VAENFLoss:
    def reconstruction_loss(self, recon_x, x):
        """Loss based on binary cross entropy between original and reconstructed input

        Args
        recon_x  -- reconstructed input
        x        -- original input

        Returns binary cross entropy between x and x_recon
        """
        return F.mse_loss(recon_x, x.view(-1, recon_x.shape[-1]), reduction='mean')

    def kld_loss(self, mu, logvar):
        """Loss based on KL-divergence"""
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def VAE_loss(self, recon_x, x, mu, logvar):
        """Variational Autoencoder loss
        Args:
        recon_x   -- reconstruction of x
        x         -- original x
        mu        -- amortized mean
        logvar    -- amortized log variance
        """
        return self.reconstruction_loss(recon_x, x) + self.kld_loss(mu, logvar) / x.size(0)

    def VAENF_loss(self, recon_x, x, mu, logvar, sum_log_det):
        """Variational Autoencoder with Normalizing Flow loss
        Args:
        recon_x      -- reconstruction of x
        x            -- original x
        mu           -- amortized mean
        logvar       -- amortized log variance
        sum_log_det  -- sum of log jacobians
        """
        return self.VAE_loss(recon_x, x, mu, logvar) - sum_log_det.mean()
