import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .vqvae import VectorQuantizedVAE
EPS = 1e-7


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, transpose=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        if transpose:
            pe[:, 0, 0::2] = torch.cos(position * div_term)
            pe[:, 0, 1::2] = torch.sin(position * div_term)
        else:
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.pe = nn.Parameter(pe, requires_grad=True)
        # self.register_parameter('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class VQA(nn.Module):
    def __init__(self, input_dim, dim, edim, vocab, out_vocab, output=64, rnn_dim=256, max_len=50, **kwargs):
        super(VQA, self).__init__()
        self.vqvae = VectorQuantizedVAE(input_dim, dim, edim, **kwargs)
        self.vocab = vocab
        self.rnn_dim = rnn_dim
        self.outlen = self.vqvae.latent_shape[1]*self.vqvae.latent_shape[2]

        self.encoder = nn.TransformerEncoder(
                            nn.TransformerEncoderLayer(
                                d_model=rnn_dim,
                                dim_feedforward=4*rnn_dim,
                                nhead=4),  # 4
                            num_layers=10)  # 8

        self.q_embed = nn.Embedding(len(self.vocab), rnn_dim,  # rnn_dim
                                        padding_idx=self.vocab.pad())

        self.q_pos_embed = PositionalEncoding(rnn_dim, max_len=max_len)  # rnn_dim

        self.img_embed = nn.Embedding(self.vqvae.n_codes, rnn_dim)  # rnn_dim
        self.img_posx_embed = PositionalEncoding(rnn_dim // 2,
                                                 max_len=self.vqvae.latent_shape[1])
        self.img_posy_embed = PositionalEncoding(rnn_dim // 2,
                                                 max_len=self.vqvae.latent_shape[2],
                                                 transpose=True)

        self.answer_proj = nn.Linear(rnn_dim, len(out_vocab))  # rnn_dim


    def init_weights(self):
        initrange = 0.1
        self.q_embed.weight.data.uniform_(-initrange, initrange)
        self.img_embed.weight.data.uniform_(-initrange, initrange)
        self.answer_proj.weight.data.uniform_(-initrange, initrange)
        self.answer_proj.bias.data.zero_()


    def forward(self, question, img, answer=None, predict=False):
        B, S = question.shape

        if len(img.shape) == 4:
            with torch.no_grad():
                _, encodings, *_ = self.vqvae.encode(img)  # B x H x W
                enc_T = encodings.view(B, -1).transpose(0, 1)  # L X B (L = HW)

        else:
            enc_T = img.transpose(0, 1)

        L, _ = enc_T.shape
        posx_embed = self.img_posx_embed.pe[np.arange(L) % self.vqvae.latent_shape[1]]
        posy_embed = self.img_posy_embed.pe[np.arange(L) // self.vqvae.latent_shape[1]]
        out_pos_embed = torch.cat((posx_embed, posy_embed), dim=-1)
        img_embed = self.img_embed(enc_T) + out_pos_embed
        img_mask = torch.zeros_like(img_embed[:, :, 0])

        question = question.transpose(0, 1)
        question_embed = self.q_pos_embed(self.q_embed(question))
        q_mask = (question == self.vocab.pad()).float()  # S X B

        qimg = torch.cat((question_embed, img_embed), dim=0)
        qimg_mask = torch.cat((q_mask, img_mask), dim=0)

        source = self.encoder(qimg,
                              src_key_padding_mask=qimg_mask.transpose(0, 1))   #  S X B X D

        logits = self.answer_proj(source[0])

        if answer is None:
            nll = 0*logits[0]
        else:
            nll = F.cross_entropy(logits, answer)

        if predict:
            return nll, logits.argmax(dim=-1)

        return nll
