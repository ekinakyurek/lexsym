import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import weights_init

class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, dim, edim, K=16, cc=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim , 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, 2*dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2*dim, 2*dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2*dim, 2*dim, 3, 1, 1),
            nn.LeakyReLU(0.2),
            ResBlock(2*dim, dim // 2),
            ResBlock(2*dim, dim // 2),
            nn.Conv2d(2*dim, 2*edim, 1, 1),
        )

        self.codebook1 = VectorQuantizerEMA(K, edim,  cc=cc, decay=decay, epsilon=epsilon)

        self.codebook2 = VectorQuantizerEMA(K, edim,  cc=cc, decay=decay, epsilon=epsilon)

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

        # self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        z_e_x1, z_e_x2 = z_e_x.chunk(2, dim=1)
        q1, e1, l1, nll1 = self.codebook1(z_e_x1)
        q2, e2, l2, nll2 = self.codebook2(z_e_x2)
        q  = q1 + q2
        e = torch.stack((e1,e2), dim=1)
        return q, e, l1+l2, nll1+nll2

    def decode(self, z_q_x):
        return self.decoder(z_q_x)

    def forward(self, x):
        z_q_x, latents, loss, nll = self.encode(x)
        x_tilde = self.decode(z_q_x)
        recon_error = F.mse_loss(x_tilde, x)
        loss += recon_error
        return loss, x_tilde, recon_error, nll, z_q_x, latents

class ResBlock(nn.Module):
    def __init__(self, dim, idim):
        super().__init__()
        self.block = nn.Sequential( #nn.ReLU(True),
            nn.Conv2d(dim, idim, 3, 1, 1, bias=False), #nn.BatchNorm2d(idim),
            nn.LeakyReLU(0.2),
            nn.Conv2d(idim, dim, 1, bias=False), #nn.BatchNorm2d(dim)
        )
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.activation(x + self.block(x))

class VectorQuantizerEMA(nn.Module):
    def __init__(self, K, dim, cc=0.25, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = dim
        self._num_embeddings = K

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = cc

        self.register_buffer('_ema_cluster_size', torch.zeros(K))
        self._ema_w = nn.Parameter(self._embedding.weight.clone())
        self._decay = decay

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

        # Use EMA to update the embedding vectors
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
        loss = q_latent_loss + self._commitment_cost * e_latent_loss


        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        # avg_probs = torch.mean(encodings, dim=0)
        # nll = -torch.log(avg_probs + 1e-10).sum()
        nll = torch.ones(1) # TODO

        # convert quantized from BHWC -> BCHW
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized, encodings, loss,  nll
