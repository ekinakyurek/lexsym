import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from .utils import weights_init
from .utils import View
from .utils import reset_parameters
from .utils import conv3x3
import numpy as np
from seq2seq import hlog


class VAE(nn.Module):
    def __init__(self, input_dim, dim, z_dim, beta=1.0, noise=None, size=(64, 64)):
        super().__init__()
        self.beta = beta
        self.zdim = z_dim
        self.size = size
        self.dim = dim

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
            hlog.log(f"pre fc latent_shape: {self.pre_latent_shape}")

        self.latent_shape = (8*dim,)
        hlog.log(f"latent_shape: {np.prod(self.latent_shape) // 2}")

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
            nn.ConvTranspose2d(dim, input_dim, 4, 2, padding=1),
        )
        self.dropout = nn.Dropout(0.3)
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
                img,
                cmd=None,
                variational=True,
                reconstruction_loss=True,
                z_bias=None,
                return_z=False,
                only_loss=False,
                ):
        if cmd is not None:
            cmd = cmd.transpose(0, 1)

        mu, logvar = self.encode(img, cmd)

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
            reconstruction_error = (self.noise(x_tilde) - self.noise(img))\
                                    .pow(2).sum().div(mu.shape[0])
            loss += reconstruction_error
        else:
            reconstruction_error = .0

        if type(kl_div) == float:
            kl_div = torch.tensor(kl_div, device=loss.device)

        if return_z:
            return z, loss, x_tilde, reconstruction_error*B, kl_div*B
        else:
            if only_loss:
                return loss, {'reconstruction_error': reconstruction_error,
                              'kl': kl_div}
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
