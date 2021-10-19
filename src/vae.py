import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from .utils import weights_init
from .utils import View
from .utils import reset_parameters
from .utils import conv3x3
import numpy as np
import math
from seq2seq import Encoder, EncDec
from absl import logging


class VAE(nn.Module):
    def __init__(self, input_dim, dim, z_dim, beta=1.0, noise=None, size=(64, 64)):
        super().__init__()
        self.beta = beta
        self.zdim = z_dim
        self.size = size
        if noise is not None:
            self.noise = noise
        else:
            self.noise = nn.Identity()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, 2*dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            conv3x3(2*dim, 2*dim),
            nn.Conv2d(2*dim, 3*dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            conv3x3(3*dim, 3*dim),
            nn.Conv2d(3*dim, 4*dim, 5, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4*dim, 2*5*dim, 3, 1, 0),
        )

        with torch.no_grad():
            mu, _ = self.encoder(torch.ones(1, 3, *size)).chunk(2, dim=1)
            self.pre_latent_shape = mu.shape[1:]
            logging.info(f"pre fc latent_shape: {self.pre_latent_shape}")

        self.latent_shape = (8*dim,)
        logging.info(f"latent_shape: {np.prod(self.latent_shape) // 2}")

        self.down_proj = nn.Sequential(nn.Flatten(),
                                       nn.Linear(2*np.prod(self.pre_latent_shape),
                                       2*np.prod(self.latent_shape)))
        self.encoder.add_module('down_proj', self.down_proj)


        self.up_proj = nn.Sequential(nn.Linear(np.prod(self.latent_shape),
                                     np.prod(self.pre_latent_shape)))

        self.view = View(-1, *self.pre_latent_shape)

        self.decoder = nn.Sequential(
            self.up_proj,
            self.view,
            nn.ConvTranspose2d(5*dim, 4*dim, 3, 1, 0),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(4*dim, 3*dim, 5, 1, 0),
            nn.LeakyReLU(0.2),
            conv3x3(3*dim, 3*dim),
            nn.ConvTranspose2d(3*dim, 2*dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            conv3x3(2*dim, 2*dim),
            nn.ConvTranspose2d(2*dim, dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, padding=(1, 1)),
        )

        self.apply(weights_init)

    def reset_decoder_parameters(self):
        self.up_proj.apply(reset_parameters)
        self.decoder.apply(reset_parameters)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        eps = torch.randn_like(std)
        return eps * std + mu

    def kl_div(self, mu, logvar):
        logvar = torch.clamp(logvar, max=3.0)
        return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum().div(mu.shape[0])

    def _log_prob(self, dist, z):
        return dist.log_prob(z).sum((2, 3, 4))

    def encode(self, x, cmd=None):
        mu, logvar = self.encoder(x).chunk(2, dim=-1)
        return mu, logvar

    def forward(self,
                x,
                cmd=None,
                variational=True,
                reconstruction_loss=True,
                z_bias=None,
                return_z=False,
                ):

        mu, logvar = self.encode(x, cmd)

        B = mu.shape[0]

        if variational:
            kl_div = self.kl_div(mu, logvar)
            if z_bias is not None:
                mu = mu + z_bias
            z = self.reparameterize(mu, logvar)
        else:
            if z_bias is not None:
                mu = mu + z_bias
            z = mu
            kl_div = .0

        x_tilde = self.decoder(z)
        loss = self.beta * kl_div

        if reconstruction_loss:
            reconstruction_error = (self.noise(x_tilde) - self.noise(x))\
                                    .pow(2).sum().div(mu.shape[0])
            loss += reconstruction_error
        else:
            reconstruction_error = .0

        if return_z:
            return z, loss, x_tilde, reconstruction_error*B, kl_div*B
        else:
            return loss, x_tilde, reconstruction_error*B, kl_div*B

    def nll(self, x, cmd=None, N=25):
        mu, logvar = self.encode(x, cmd)
        q_z = Normal(mu, logvar.mul(0.5).exp())
        p_z = Normal(torch.zeros_like(mu), torch.ones_like(mu))
        z = q_z.sample((N,))
        x_tilde = self.decoder(z.view(-1, *mu.shape[1:])).view(N,*x.shape)
        pxz = Normal(x_tilde, torch.ones_like(x_tilde))
        logpx = self._log_prob(pxz, x) + self._log_prob(p_z,z) - self._log_prob(q_z, z)
        return -((logpx.logsumexp(dim=0) - np.log(N)).sum(0))

    def sample(self, B=1, cmd=None, z_bias=None):
        mu = torch.zeros(B, np.prod(self.latent_shape))\
                            .to(self.encoder[0].weight.device)
        if z_bias is not None:
            mu = mu + z_bias
        p_z = Normal(mu, torch.ones_like(mu))
        z = p_z.sample()
        x_tilde = self.decoder(z)
        return x_tilde, z


class CVAE(nn.Module):
    def __init__(self, input_dim, dim, z_dim, vocab, rnn_dim=512, beta=1.0, noise=None, size=(64,64)):
        super().__init__()
        # pretrained vae
        self.vae = VAE(input_dim, dim, z_dim, beta=beta, noise=noise, size=size)

        self.lang_encoder = Encoder(vocab,
                                    rnn_dim,
                                    rnn_dim,
                                    2,  # n_layers
                                    dropout=0.1,
                                    bidirectional=False)

        self.proj = nn.Linear(rnn_dim, 2*math.prod(self.vae.latent_shape))

    def forward(self, x, cmd):
        _, (h, _) = self.lang_encoder(cmd)
        hidden = h[-1]
        params1 = self.proj(hidden)
        params2 = self.vae.encoder(x)
        return (params1.flatten()-params2.flatten()).pow(2).sum().div(params1.shape[0])
        # return (z_rnn.flatten()-z_vae.flatten()).pow(2).sum().div(z_rnn.shape[0])

    def predict(self, cmd):
        _, (h, _) = self.lang_encoder(cmd)
        hidden = h[-1]
        mu, logvar = self.proj(hidden).chunk(2, dim=1)
        mu = mu.view(mu.shape[0], *self.vae.latent_shape)
        logvar = logvar.view(logvar.shape[0], *self.vae.latent_shape)
        z_rnn = self.vae.reparameterize(mu, logvar)
        x_tilde = self.vae.decoder(z_rnn)
        return x_tilde
