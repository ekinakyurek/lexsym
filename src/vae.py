import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from .utils import weights_init
import numpy as np
import math
import pdb
from seq2seq import Encoder, EncDec

class VAE(nn.Module):
    def __init__(self, input_dim, dim, z_dim, beta=1.0, noise=None):
        super().__init__()
        self.beta = beta
        self.zdim = z_dim

        if noise is not None:
            self.noise = noise
        else:
            self.noise = nn.Identity()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 5, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(dim, z_dim * 2, 3, 1, 0),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, dim, 3, 1, 0),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 5, 1, 0),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1)
        )

        self.apply(weights_init)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        eps = torch.randn_like(std)
        return eps * std + mu

    def kl_div(self, mu, logvar):
        return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum().div(mu.shape[0])

    def _log_prob(self, dist, z):
        return dist.log_prob(z).sum((2,3,4))

    def feats(self,x,cmds=None):
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        return self.reparameterize(mu, logvar)

    def nll(self, x, cmd=None, N=25):
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        q_z = Normal(mu, logvar.mul(0.5).exp())
        p_z = Normal(torch.zeros_like(mu), torch.ones_like(mu))
        z = q_z.sample((N,))
        x_tilde = self.decoder(z.view(-1,*mu.shape[1:])).view(N,*x.shape)
        pxz = Normal(x_tilde, torch.ones_like(x_tilde))
        logpx = self._log_prob(pxz,x) + self._log_prob(p_z,z) - self._log_prob(q_z, z)
        return -((logpx.logsumexp(dim=0) - np.log(N)).sum(0))


    def forward(self, x, cmd=None):
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        kl_div = self.kl_div(mu, logvar)
        sample  = self.reparameterize(mu, logvar)
        x_tilde = self.decoder(sample)
        loss_recons = (self.noise(x_tilde) - self.noise(x)).pow(2).sum().div(mu.shape[0])
        loss = loss_recons + self.beta * kl_div
        return loss, x_tilde, loss_recons*mu.shape[0], kl_div*mu.shape[0]

    def sample(self, B=1, cmd=None):
        mu = torch.zeros(B,self.zdim,2,2).to(self.encoder[0].weight.device)
        p_z = Normal(mu, torch.ones_like(mu))
        z = p_z.sample()
        x_tilde = self.decoder(z)
        return x_tilde, z


class CVAE(nn.Module):
    def __init__(self, input_dim, dim, z_dim, rnn_dim, vocab, beta=1.0, noise=None):
        super().__init__()
        #pretrained vae
        self.vae = VAE(input_dim, dim, z_dim, beta=beta, noise=noise)

        self.lang_encoder = Encoder(vocab,
                                    rnn_dim,
                                    rnn_dim,
                                    2,#n_layers
                                    dropout=0.1)

        self.proj = nn.Linear(2*rnn_dim,2*2*z_dim)

    def forward(self, x, cmd):
        outputs, _ = self.lang_encoder(cmd)
        z_rnn = self.proj(outputs.mean(dim=0))
        z_vae = self.vae.feats(x)
        return (z_rnn.flatten()-z_vae.flatten()).pow(2).sum().div(z_rnn.shape[0])

    def predict(self, cmd):
        outputs, _ = self.lang_encoder(cmd)
        z_rnn = self.proj(outputs.mean(dim=0))
        x_tilde = self.vae.decoder(z_rnn.view(z_rnn.shape[0],-1,2,2))
        return x_tilde
