import torch.nn as nn
import numpy as np
import math
from seq2seq import Encoder
from vae import VAE


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
