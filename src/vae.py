import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from .utils import weights_init

class VAE(nn.Module):
    def __init__(self, input_dim, dim, z_dim, beta=1.0, noise=None):
        super().__init__()
        self.beta = beta
        if noise is not None:
            sel.noise = noise
        else:
            sel.noise = nn.Identity()

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

        # self.apply(weights_init)

    def forward(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        q_z_x = Normal(mu, logvar.mul(.5).exp())
        p_z = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        kl_div = kl_divergence(q_z_x, p_z).sum(1).mean()
        x_tilde = self.decoder(q_z_x.rsample())
        loss_recons = F.mse_loss(sel.noise(x_tilde), sel.noise(x), size_average=False).div(x.size(0))
        loss = loss_recons + self.beta * kl_div
        return loss, x_tilde, loss_recons, kl_div
