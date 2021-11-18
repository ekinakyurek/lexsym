import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import weights_init, top_k_logits, fig2tensor
import math
import numpy as np
import json
from torch.distributions.normal import Normal
from seq2seq import TransformerDecoderv2, TransformerDecoderLayerv2
import operator
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import torchvision.transforms.functional as TF
from absl import logging, flags

EPS = 1e-7

FLAGS = flags.FLAGS

flags.DEFINE_integer('n_codes', default=10,
                     help='Sets number of codes in vqvae.')

flags.DEFINE_float('beta', default=1.0,
                   help='Sets beta parameter in beta vae.')

flags.DEFINE_float('commitment_cost', default=0.25,
                   help='Sets commitment lost in vqvae')

flags.DEFINE_float('epsilon', default=1e-5,
                   help='Sets epsilon value in VQVAE.')


class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, dim, edim, n_codes=16, cc=0.25, decay=0.99, epsilon=1e-5, beta=1.0, cmdproc=False, size=(64,64)):
        super().__init__()

        self.h_dim = dim
        self.l_dim = edim

        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, 2*dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2*dim, 2*dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2*dim, 2*dim, 3, 1, 1),
            nn.LeakyReLU(0.2),
            ResBlock(2*dim, dim // 2),
            ResBlock(2*dim, dim // 2),
            nn.Conv2d(2*dim, edim, 1, 1),
        )

        with torch.no_grad():
            mu = self.encoder(torch.ones(1,3,*size))
            self.latent_shape= (self.l_dim,mu.shape[2], mu.shape[3])
            logging.info(f"latent_shape: {self.latent_shape}")

        self.codebook1 = VectorQuantizerEMA(n_codes, self.latent_shape, cc=cc, decay=decay, epsilon=epsilon, beta=beta, cmdproc=cmdproc)
        #
        # self.codebook2 = VectorQuantizerEMA(n_codes, edim,  cc=cc, decay=decay, epsilon=epsilon)
        #
        # self.codebook3 = VectorQuantizerEMA(n_codes, edim,  cc=cc, decay=decay, epsilon=epsilon)
        #
        # self.codebook4 = VectorQuantizerEMA(n_codes, edim,  cc=cc, decay=decay, epsilon=epsilon)

        self.decoder = nn.Sequential(
            nn.Conv2d(edim, 2*dim, 3, 1, 1),
            nn.LeakyReLU(0.2),
            ResBlock(2*dim, dim // 2),
            ResBlock(2*dim, dim // 2),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(2*dim, dim, 4, 1, 2),
            nn.LeakyReLU(0.2),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(dim, dim, 4, 1, 1),
            nn.LeakyReLU(0.2),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(dim, input_dim, 3, 1),
        )

        self.apply(weights_init)

    def encode(self, x, cmds=None):
        z_e_x = self.encoder(x)
        return self.codebook1(z_e_x, cmds)
        # z_e_x1, z_e_x2, z_e_x3, z_e_x4 = z_e_x.chunk(4, dim=1)
        # cmds = cmds.view(3,3,-1) # 3 x 3 x B
        # q1, e1, l1, nll1 = self.codebook1(z_e_x1, cmds[:,0,:])
        # q2, e2, l2, nll2 = self.codebook2(z_e_x2, cmds[:,1,:])
        # q3, e3, l3, nll3 = self.codebook3(z_e_x3, cmds[:,2,:])
        # q4, e4, l4, nll4 = self.codebook4.forward2(z_e_x4)
        # q  = (q1 + q2 + q3 + q4) / 3
        # e = torch.stack((e1,e2,e3,e4), dim=1)
        # return q, e, (l1+l2+l3+l4) / 4, (nll1+nll2+nll3+nll4) / 4

    def sample(self, B=1, cmd=None):
        z_q_x, encodings = self.codebook1.sample(B=B, cmd=cmd)
        x_tilde = self.decode(z_q_x)
        return x_tilde, z_q_x, encodings  # FIX: convert back to 'encodings'

    def decode(self, z_q_x, cmds=None):
        return self.decoder(z_q_x)

    def forward(self,
                img,
                cmd=None,
                reconstruction_loss=True,
                return_z=False,
                only_loss=False,
                ):
        if cmd is not None:
            cmd = cmd.transpose(0, 1)

        z_q_x, latents, loss, nll = self.encode(img, cmd)
        x_tilde = self.decode(z_q_x, cmd)
        reconstruction_error = F.mse_loss(x_tilde, img, reduction='sum')

        if reconstruction_loss:
            loss += (reconstruction_error / math.prod(img.shape))

        nll = torch.tensor(nll, device=loss.device)
        if return_z:
            return z_q_x, loss, x_tilde, reconstruction_error, nll, latents
        else:
            if only_loss:
                return loss, {'reconstruction_error': reconstruction_error,
                              'nll': nll}
            return loss, x_tilde, reconstruction_error, nll, z_q_x, latents

    def _log_prob(self, dist, z):
        return dist.log_prob(z).sum((1, 2, 3))

    @property
    def n_codes(self):
        return self.codebook1._num_embeddings

    def nll(self, img, cmds=None):
        z_q_x, latents, loss, nll = self.encode(img, cmds)
        x_tilde = self.decode(z_q_x, cmds)
        pxz = Normal(x_tilde, torch.ones_like(x_tilde))
        return -self._log_prob(pxz, img).sum() + nll # this is single sample estimate, so it's lower bound


class ResBlock(nn.Module):
    def __init__(self, dim, idim):
        super().__init__()
        self.block = nn.Sequential(  # nn.ReLU(True),
            nn.Conv2d(dim, idim, 3, 1, 1, bias=False),  # nn.BatchNorm2d(idim),
            nn.LeakyReLU(0.2),
            nn.Conv2d(idim, dim, 1, bias=False),  # nn.BatchNorm2d(dim)
        )
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.activation(x + self.block(x))

class VectorQuantizerEMA(nn.Module):
    def __init__(self, n_codes, latent_shape, cc=0.25, decay=0.99, epsilon=1e-5, beta=1.0, cmdproc=False):
        super(VectorQuantizerEMA, self).__init__()

        self.cmdproc = cmdproc
        self.beta = beta
        self._latent_shape = latent_shape

        self._embedding_dim = latent_shape[0]
        self._num_embeddings = n_codes

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = cc

        self.register_buffer('_ema_cluster_size', torch.zeros(n_codes))
        self._ema_w = nn.Parameter(self._embedding.weight.clone())
        self._decay = decay

        self._epsilon = epsilon

    def forward(self, inputs, cmds=None):
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
        quantized = self._embedding(encodings)
        # Inefficient version of the above after materializing one hot embeddings:
        # quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Language supervision
        if cmds is not None and self.cmdproc:
            cmd_loss = self.command_loss(cmds, distances, input_shape)
        else:
            cmd_loss = 0

        # # Use EMA to update the embedding vectors
        if self.training:
            one_hot = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
            one_hot.scatter_(1, encoding_indices.unsqueeze(1), 1)
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * one_hot.sum(dim=0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)

            dw = one_hot.t() @ flatten
            self._ema_w.data = self._ema_w * self._decay + (1 - self._decay) * dw

            self._embedding.weight.data = self._ema_w / self._ema_cluster_size.unsqueeze(1)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss + cmd_loss
        loss *= self.beta


        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        nll = -(torch.numel(encodings) * math.log(self._num_embeddings))
        #nll = -torch.log(avg_probs + 1e-10).sum()
        #nll = torch.ones(1) # TODO

        # convert quantized from BHWC -> BCHW
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized, encodings, loss,  nll

    def command_loss(self, cmds, distances, input_shape, lamda=0.01):
        dists = distances.view(input_shape[0], input_shape[1] * input_shape[2], self._num_embeddings)  # [B, HW, n_codes]
        num_objs = torch.count_nonzero(cmds, dim=0)  # B
        loss = 0.
        for (k, inst) in enumerate(dists.split(1, dim=0)):  #
            regions = inst.squeeze(0).chunk(num_objs[k], dim=0)
            for (j, region) in enumerate(regions):
                loss += region[:, cmds[j, k]].mean()
        return lamda * (loss / input_shape[0])

    def sample(self, B=1, cmd=None):
        encodings = np.random.choice(np.arange(self._num_embeddings), (B, *self._latent_shape[1:]))
        encodings = torch.from_numpy(encodings).to(self._embedding.weight.device)
        quantized = self._embedding(encodings)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized, encodings
