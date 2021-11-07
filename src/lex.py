import torch
from torch import nn
import torch.nn.functional as F
import math
from absl import logging
from . import vae
from .vqvae import VectorQuantizedVAE
from .utils import reset_parameters
from .utils import weights_init
from .utils import conv3x3
from .utils import View
from .utils import LambdaLayer
import numpy as np

EPS = 1e-5

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
    def __init__(self,
                 n_downsample,
                 d_latent,
                 d_embed,
                 latent_shape=None,
                 decoder_dim=None,
                 vae_z_len=None,
                 ):

        super().__init__()

        downsample_layers = []
        upsample_layers = []

        downsample_projections = []
        upsample_projections = []

        downsample_sizes = (
            [(4, d_latent)]
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
            # upsample_block.append(u_positional)
            upsample_layers.append(nn.Sequential(*upsample_block))

        self.downsample_layers = nn.ModuleList(downsample_layers)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.downsample_projections = nn.ModuleList(downsample_projections)
        self.upsample_projections = nn.ModuleList(upsample_projections)

        if latent_shape is not None and decoder_dim is not None:
            self.embed_decoder = self.get_decoder_layer(d_embed,
                                                        decoder_dim,
                                                        3,
                                                        latent_shape,
                                                        vae_z_len)
        else:
            self.embed_decoder = None

        self.dropout = nn.Dropout(0.3)
        # self.fulldrop = nn.Dropout(0.9)

    def get_decoder_layer(self, d_embed, dim, input_dim, latent_shape, vae_z_len):

        up_proj = nn.Sequential(nn.Linear(d_embed + vae_z_len,
                                          np.prod(latent_shape)))

        view = View(-1, *latent_shape)

        pre_layer = nn.Sequential(
                        up_proj,
                        view,
                        Positional()
                    )
        concat_layer = LambdaLayer(
            lambda x, y: torch.cat((x, F.interpolate(y, size=(x.shape[2], x.shape[3]))), dim=1))

        after_layer = nn.Sequential(
                        nn.ConvTranspose2d(5*dim+1, 4*dim, 3, 1, 0),
                        nn.LeakyReLU(0.2),
                        nn.ConvTranspose2d(4*dim, 3*dim, 5, 1, 0),
                        nn.LeakyReLU(0.2),
                        nn.ConvTranspose2d(3*dim, 2*dim, 4, 2, 1),
                        nn.LeakyReLU(0.2),
                        nn.ConvTranspose2d(2*dim, dim, 4, 2, 1),
                        nn.LeakyReLU(0.2),
                        nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
                    )
        return nn.ModuleList([pre_layer, concat_layer, after_layer])

    def forward(self, img, att_map, embedding, z=None):
        img = self.dropout(img)
        features = torch.cat((img, att_map), dim=1)
        for i in range(len(self.downsample_layers)):
            features = self.dropout(features) +\
             self.downsample_projections[i](embedding)[..., None, None]
            features = self.downsample_layers[i](features)

        for i in range(len(self.upsample_layers)):
            features = self.dropout(features) +\
             self.upsample_projections[i](embedding)[..., None, None]
            features = self.upsample_layers[i](features)

        if self.embed_decoder is not None and z is not None:
            embedding = torch.cat((embedding,  F.dropout(z, 0.9)), dim=-1)

            features = features + self.embed_decoder[2](
                            self.embed_decoder[1](
                                self.embed_decoder[0](embedding), att_map))

        return features

class FilterModel(nn.Module):
    def __init__(self,
                 vocab,
                 n_downsample,
                 n_latent,
                 n_steps,
                 att_loss_weight=0.0001,
                 vae=None,
                 text_conditional=False,
                 append_z=False):

        super().__init__()

        self.vocab = vocab
        self.n_latent = n_latent
        self.n_steps = n_steps
        self.h_dim = 2 * n_latent
        self.att_loss_weight = att_loss_weight
        self.vae = vae
        self.text_conditional = text_conditional
        self.append_z = append_z
        self.filter_embed_dim = self.h_dim

        if vae is not None and text_conditional:
            self.cond_emb = nn.Linear(n_latent,
                                      math.prod(self.vae.latent_shape))

            if append_z:
                self.vae_proj = nn.Linear(math.prod(self.vae.latent_shape),
                                          n_latent)
                self.filter_embed_dim = self.h_dim + n_latent

        self.emb = nn.Embedding(len(vocab), self.h_dim, padding_idx=vocab.pad())

        # self.rnn = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=self.h_dim,
        #                                dim_feedforward=4*self.h_dim,
        #                                nhead=4),
        #     num_layers=4)

        # self.pos_embed = nn.Parameter(torch.zeros(51, 1, self.h_dim))

        self.rnn = nn.LSTM(self.h_dim, self.h_dim, 1, bidirectional=False)

        self.proj = nn.Linear(self.h_dim, n_latent)

        self.lookups = nn.Linear(n_latent, n_steps * n_latent)

        self.att_featurizers = nn.ModuleList([nn.Sequential(
            nn.ReplicationPad2d(2),
            nn.Conv2d(3, n_latent, 5, 1),
            nn.LeakyReLU(0.2),
            Positional(),
            nn.Conv2d(n_latent, n_latent, 1, 1),
            nn.LeakyReLU(0.2),
        ) for i in range(n_steps)])

        if self.vae is not None:
            self.filter = ImageFilter(n_downsample,
                                      n_latent,
                                      self.filter_embed_dim,
                                      latent_shape=self.vae.pre_latent_shape,
                                      vae_z_len=np.prod(self.vae.latent_shape),
                                      decoder_dim=self.vae.dim)
        else:
            self.filter = ImageFilter(n_downsample, n_latent, self.filter_embed_dim)

        self.loss = nn.MSELoss(reduction='sum')

        self.dropout = nn.Dropout(0.3)

        self.apply(weights_init)

    def reset_decoder_parameters(self):
        if self.vae is not None:
            self.vae.reset_decoder_parameters()
        self.filter.apply(reset_parameters)
        self.rnn.reset_parameters()
        self.att_featurizers.apply(reset_parameters)
        self.proj.reset_parameters()
        self.lookups.reset_parameters()
        self.emb.reset_parameters()


    def forward(self, cmd, img, test=False, prior=False, variational=True):
        # Needed in data parallel is there a way, is there a way to fix?
        if isinstance(self.rnn, nn.LSTM):
            self.rnn.flatten_parameters()
        cmd_t = cmd.transpose(0, 1)
        embedding = self.emb(cmd_t)
        #pos_embed = self.pos_embed[:cmd_t.shape[0], :, :]
        encoding, _ = self.rnn(embedding)
        encoding = self.proj(self.dropout(encoding))

        batch_size = cmd.shape[0]

        if self.vae is not None:
            if self.text_conditional:
                # sum over sequence dimension
                z_bias = self.cond_emb(encoding.sum(dim=0))
            else:
                z_bias = None

            if prior:
                init, z = self.vae.sample(B=batch_size, z_bias=z_bias)
                kl_loss = .0
            else:
                z, kl_loss, init, *_ = self.vae(img,
                                                variational=variational,
                                                reconstruction_loss=False,
                                                z_bias=z_bias,
                                                return_z=True)
                kl_loss *= batch_size

            if self.append_z:
                z = self.vae_proj(z).unsqueeze(0).expand(embedding.shape[0], -1, -1)
                embedding = torch.cat((embedding, z), dim=-1)
                # encoding = torch.cat((encoding, z)), dim=-1)
        else:
            init = torch.zeros_like(img)
            kl_loss = .0

        lookups = self.lookups(encoding.mean(dim=0)).view(-1,
                                                          self.n_steps,
                                                          self.n_latent)


        results = []
        attentions = []
        text_attentions = []

        init = self.dropout(init).mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)

        results.append(init)

        pred_loss = 0
        for i in range(self.n_steps):
            old_result = results[-1]

            lookup = lookups[None, :, i, :]
            text_attention = (lookup * encoding).sum(dim=2, keepdim=True)
            text_attention = F.gumbel_softmax(text_attention,
                                              tau=0.2,
                                              hard=False,
                                              dim=0)

            enc_attended = (encoding * text_attention).sum(dim=0)
            emb_attended = (embedding * text_attention).sum(dim=0)

            att_features = self.att_featurizers[i](old_result)

            att_map = torch.einsum('bchw,bc->bhw', att_features, enc_attended).unsqueeze(dim=1)

            att_map = torch.sigmoid(att_map) + EPS

            transformed = self.filter(old_result, att_map, emb_attended, z)

            insertion = att_map * transformed

            new_result = insertion + (1 - att_map) * old_result

            pred_loss += self.loss(insertion, att_map * img)

            attentions.append(att_map)
            results.append(new_result)
            if test:
                text_attentions.append(text_attention.transpose(0, 1))


        reconstruction = results[-1]
        pred_loss += self.loss(reconstruction, img)
        #att_loss = torch.stack(attentions).sum()
        att_loss = 0*pred_loss

        with torch.no_grad():
            scalars = {'pred_loss': pred_loss / batch_size,
                       'att_loss': att_loss / batch_size,
                       'kl_loss': torch.tensor(kl_loss / batch_size,
                                               device=pred_loss.device)}

        loss = (pred_loss + self.att_loss_weight * att_loss + kl_loss) / batch_size

        if test:
            return loss, scalars, reconstruction, results, attentions, text_attentions
        else:
            return loss, scalars
