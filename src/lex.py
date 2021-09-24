import torch
from torch import nn
import torch.nn.functional as F
import math
from absl import logging
from . import vae

class Positional(nn.Module):
    def __init__(self):
        super().__init__()
        self._cache = {}
        self._max_cache_size = 5

    def get_positional_encoding_2d(self, d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if (d_model, height, width) not in self._cache:
            if len(self._cache) == self._max_cache_size:
                raise ValueError('cache size is exceeded for positional encodings')
            if d_model % 4 != 0:
                raise ValueError("Cannot use sin/cos positional encoding with "
                                 "odd dimension (got dim={:d})".format(d_model))
            pe = torch.zeros(d_model, height, width)
            # Each dimension use half of d_model
            d_model = int(d_model / 2)
            div_term = torch.exp(torch.arange(0., d_model, 2) *
                                 -(math.log(10000.0) / d_model))
            pos_w = torch.arange(0., width).unsqueeze(1)
            pos_h = torch.arange(0., height).unsqueeze(1)
            pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
            pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
            pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
            pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
            self._cache[(d_model, height, width)] = pe

        return self._cache[(d_model, height, width)]

    def forward(self, x):
        return (
            x
            + self.get_positional_encoding_2d(*x.shape[1:4]).to(x.device)
        )

class ImageFilter(nn.Module):
    def __init__(self, n_downsample, d_latent, d_embed):
        super().__init__()

        downsample_layers = []
        upsample_layers = []

        downsample_projections = []
        upsample_projections = []

        downsample_sizes = (
            [(3, d_latent)]
            + [(d_latent, d_latent) for _ in range(n_downsample-1)]
        )

        for (in_size, out_size) in downsample_sizes:
            downsample_projections.append(nn.Linear(d_embed, in_size, bias=False))

            downsample_block = []
            downsample_block.append(nn.ReplicationPad2d(1))
            downsample_block.append(nn.Conv2d(in_size, out_size, 3, 2))
            # downsample_block.append(nn.BatchNorm2d(out_size, 1e-3))
            downsample_block.append(nn.LeakyReLU(0.2))
            downsample_block.append(Positional())
            downsample_layers.append(nn.Sequential(*downsample_block))


        upsample_sizes = (
            [(d_latent, d_latent) for _ in range(n_downsample-1)]
            + [(d_latent, 3)]
        )

        for i, (in_size, out_size) in enumerate(upsample_sizes):
            upsample_projections.append(nn.Linear(d_embed, in_size, bias=False))

            upsample_block = []
            upsample_block.append(nn.UpsamplingNearest2d(scale_factor=2))
            upsample_block.append(nn.ReplicationPad2d(1))
            upsample_block.append(nn.Conv2d(in_size, out_size, 3, 1))
            # upsample_block.append(nn.BatchNorm2d(out_size, 1e-3))
            # TODO last should be sigmoid?
            if i == n_downsample - 1:
                # upsample_block.append(nn.Sigmoid())
                pass
            else:
                upsample_block.append(nn.LeakyReLU(0.2))
            upsample_layers.append(nn.Sequential(*upsample_block))

        self.downsample_layers = nn.ModuleList(downsample_layers)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.downsample_projections = nn.ModuleList(downsample_projections)
        self.upsample_projections = nn.ModuleList(upsample_projections)

    def forward(self, img, embedding):
        features = img
        for i in range(len(self.downsample_layers)):
            features = features + self.downsample_projections[i](embedding)[..., None, None]
            features = self.downsample_layers[i](features)

        for i in range(len(self.upsample_layers)):
            features = features + self.upsample_projections[i](embedding)[..., None, None]
            features = self.upsample_layers[i](features)

        return features

class FilterModel(nn.Module):
    def __init__(self, vocab, n_downsample, n_latent, n_steps, vae=None):
        super().__init__()

        self.vocab = vocab
        self.n_latent = n_latent
        self.n_steps = n_steps
        self.h_dim = 128

        self.emb = nn.Embedding(len(vocab), self.h_dim)
        self.rnn = nn.LSTM(self.h_dim, n_latent, 1, bidirectional=True)
        self.proj = nn.Linear(n_latent * 2, n_latent)

        self.lookups = nn.Linear(n_latent, n_steps * n_latent)

        self.att_featurizers = nn.ModuleList([nn.Sequential(
            nn.ReplicationPad2d(2),
            nn.Conv2d(3, n_latent, 5, 1),
            nn.LeakyReLU(0.2),
            Positional(),
            nn.Conv2d(n_latent, n_latent, 1, 1),
            nn.LeakyReLU(0.2),
        ) for i in range(n_steps)])

        self.filter = ImageFilter(n_downsample, n_latent, self.h_dim)

        self.vae = vae
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, cmd, img, test=False, prior=False):
        self.rnn.flatten_parameters()  # Needed in data parallel is there a way, is there a way to fix?
        embedding = self.emb(cmd.transpose(0, 1))
        encoding, _ = self.rnn(embedding)
        encoding = self.proj(encoding)
        lookups = self.lookups(encoding.mean(dim=0)).view(-1, self.n_steps, self.n_latent)

        if self.vae is not None:
            if prior:
                init, z = self.vae.sample(B=cmd.shape[0])
                kl_loss = .0
            else:
                mu, logvar = self.vae.encode(img)
                z = self.vae.reparameterize(mu, logvar)
                kl_div = self.vae.kl_div(mu, logvar)
                init = self.vae.decoder(z)
                kl_loss = self.vae.beta * kl_div
        else:
            init = torch.zeros_like(img)
            kl_loss = .0

        results = []
        attentions = []
        text_attentions = []
        results.append(init)
        for i in range(self.n_steps):
            old_result = results[-1]

            text_attention = (lookups[None, :, i, :] * encoding).sum(dim=2,
                    keepdim=True)
            text_attention = F.gumbel_softmax(text_attention, hard=False, dim=0)

            # text_attention = torch.zeros_like(text_attention)
            # i_attend = 1 if i < self.n_steps // 2 else 0
            # text_attention[i_attend, ...] = 1

            enc_attended = (encoding * text_attention).sum(dim=0)
            emb_attended = (embedding * text_attention).sum(dim=0)

            att_features = self.att_featurizers[i](old_result)
            # logging.info(f"att_features before permute: {att_features.shape}")
            # att_features = att_features.permute(0, 2, 3, 1)
            # logging.info(f"att_features after permute: {att_features.shape}")
            # logging.info(f"enc_attended: {enc_attended.shape}")
            att_map = torch.einsum('bchw,bc->bhw', att_features, enc_attended)
            att_map = torch.sigmoid(att_map).unsqueeze(dim=1)
            # att_map = (enc_attended[:, None, None, :] * att_features).sum(dim=3, keepdim=True)

            # att_map = torch.sigmoid(att_map).permute(0, 3, 1, 2)

            transformed = self.filter(old_result, emb_attended)
            new_result = att_map * transformed + (1 - att_map) * old_result
            attentions.append(att_map)
            if test:
                text_attentions.append(text_attention.transpose(0, 1))
            results.append(new_result)

        pred_loss = self.loss(results[-1], img).div(init.shape[0])
        att_loss = torch.stack(attentions).sum().div(init.shape[0])
        scalars = {'pred_loss': pred_loss,
                   'att_loss': att_loss,
                   'kl_loss': torch.tensor(kl_loss, device=pred_loss.device)}
        loss = pred_loss + 0.0001 * att_loss + kl_loss
        if test:
            return loss, scalars, results[-1], results, attentions, text_attentions
        else:
            return loss, scalars
