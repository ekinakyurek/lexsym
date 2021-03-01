import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import pdb

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)

class ResBlock(nn.Module):
    def __init__(self, dim, idim):
        super().__init__()
        self.block = nn.Sequential( #nn.ReLU(True),
            nn.Conv2d(dim, idim, 3, 1, 1, bias=False), #nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(idim, dim, 1, bias=False), #nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return F.relu(x + self.block(x))


class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, dim, edim, std=1.0, K=16, cc=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.var = nn.Parameter(std**2,requires_grad=False)

        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim , 4, 2, 1), #nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 2*dim, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(2*dim, 2*dim, 3, 1, 1),
            nn.ReLU(True),
            ResBlock(2*dim, dim // 2),
            ResBlock(2*dim, dim // 2),
            nn.Conv2d(2*dim, edim, 1, 1),
        )

        self.codebook = VectorQuantizerEMA(K, edim,  cc=cc, decay=decay, epsilon=epsilon)

        self.decoder = nn.Sequential(
            nn.Conv2d(edim, 2*dim, 3, 1, 1),
            nn.ReLU(True),
            ResBlock(2*dim, dim // 2),
            ResBlock(2*dim, dim // 2), #nn.ReLU(True),
            nn.ConvTranspose2d(2*dim, dim, 4, 2, 1),#nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
        )

        # self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        return self.codebook(z_e_x)

    def decode(self, z_q_x):
        return self.decoder(z_q_x)

    def forward(self, x):
        z_q_x, latents, loss, nll = self.encode(x)
        x_tilde = self.decode(z_q_x)
        recon_error = F.mse_loss(x_tilde, x)
        loss += recon_error
        return loss, x_tilde, recon_error, nll, z_q_x, latents


class VectorQuantizerEMA(nn.Module):
    def __init__(self, K, dim, cc=0.25, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = dim
        self._num_embeddings = K

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = cc

        # self.register_buffer('_ema_cluster_size', torch.zeros(K))
        # self._ema_w = nn.Parameter(torch.Tensor(K, self._embedding_dim))
        # self._ema_w.data.normal_()
        # self._decay = decay

        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flatten = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self._embedding.weight.t()
            + self._embedding.weight.pow(2).sum(1, keepdim=True).t()
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = encoding_indices.view(input_shape[0:3])
        # Quantize and unflatten
        # quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        quantized = self._embedding(encodings)

        # # Use EMA to update the embedding vectors
        # if self.training:
        #
        #     self._ema_cluster_size = self._ema_cluster_size * self._decay + \
        #                              (1 - self._decay) * torch.sum(encodings, 0)
        #
        #     # Laplace smoothing of the cluster size
        #     n = torch.sum(self._ema_cluster_size.data)
        #     self._ema_cluster_size = (
        #         (self._ema_cluster_size + self._epsilon)
        #         / (n + self._num_embeddings * self._epsilon) * n)
        #
        #     dw = torch.matmul(encodings.t(), flat_input)
        #     self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
        #
        #     self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss


        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        # avg_probs = torch.mean(encodings, dim=0)
        # nll = -torch.log(avg_probs + 1e-10).sum()
        nll = torch.ones(1)

        # convert quantized from BHWC -> BCHW
        return quantized.permute(0, 3, 1, 2).contiguous(), encodings, loss,  nll