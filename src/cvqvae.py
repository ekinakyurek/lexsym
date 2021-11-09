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
from absl import logging
from vqvae import VectorQuantizedVAE
EPS = 1e-7


class CVQVAE(nn.Module):
    def __init__(self, input_dim, dim, edim, vocab, rnn_dim=256, max_len=50, **kwargs):
        super(CVQVAE, self).__init__()
        self.vqvae = VectorQuantizedVAE(input_dim, dim, edim, **kwargs)
        self.vocab = vocab
        self.rnn_dim = rnn_dim
        self.drop = nn.Dropout(0.2)
        self.highdrop = nn.Dropout(0.5)
        self.outlen = self.vqvae.latent_shape[1]*self.vqvae.latent_shape[2]
        # Encoder Transformer
        self.inp_embed = nn.Embedding(len(self.vocab), rnn_dim, padding_idx=self.vocab.pad())
        self.inp_pos_embed = nn.Parameter(torch.zeros(max_len, 1 ,rnn_dim))
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=rnn_dim, dim_feedforward=4*rnn_dim, nhead=4), num_layers=4)
        # Decoder Transformer
        self.out_embed = nn.Embedding(self.vqvae.n_codes, rnn_dim)
        self.out_posx_embed = nn.Parameter(torch.zeros(self.vqvae.latent_shape[1], 1, rnn_dim // 2))
        self.out_posy_embed = nn.Parameter(torch.zeros(self.vqvae.latent_shape[2], 1, rnn_dim // 2))
        self.start_embed = nn.Parameter(torch.zeros(1, 1, rnn_dim))
        self.decoder = TransformerDecoderv2(TransformerDecoderLayerv2(d_model=rnn_dim, dim_feedforward=4*rnn_dim, nhead=4), return_attentions=True, num_layers=4)
        # Final Projection
        self.seq_picker = nn.Linear(rnn_dim, 1)  # TODO: 3
        self.proj = nn.Linear(rnn_dim,self.vqvae.n_codes)
        self.copy_criterion = nn.BCEWithLogitsLoss()
        self.set_tgt_mask()
        self.set_self_copy()

    def get_src_lexicon(self, lexfile):
        with open(lexfile, "r") as f:
            matchings = json.load(f)
        self.matchings = matchings
        self.rev_matchings = {}
        for (k, stats) in self.matchings.items():
             code = max(stats.items(), key=operator.itemgetter(1))[0]
             self.rev_matchings[int(code)] = k
        lexicon = np.zeros((len(self.vocab), self.vqvae.n_codes))
        src_keys = list(self.vocab._contents.keys())
        for (k, matching) in matchings.items():
            ki = src_keys.index(k)
            for (v, c) in matching.items():
                vi = int(v)
                lexicon[ki, vi] = 10000
        lexicon = torch.from_numpy(lexicon).float()
        # pdb.set_trace()
        return torch.softmax(lexicon, dim=1)

    def set_tgt_mask(self):
        mask = torch.tril(torch.ones(self.outlen, self.outlen)) == 0.0
        self.register_buffer("tgt_mask", mask)

    def set_self_copy(self):
        self.self_copy_proj = nn.Embedding(self.vqvae.n_codes, self.vqvae.n_codes)
        self.self_copy_proj.weight.data = torch.eye(self.vqvae.n_codes)
        self.self_copy_proj.weight.requires_grad = False

    def set_src_copy(self, lexicon, requires_grad=False):
        if torch.is_tensor(lexicon):
            self.src_copy_proj = nn.Embedding(lexicon.shape[0], lexicon.shape[1])
            self.src_copy_proj.weight.data = lexicon
            self.src_copy_proj.weight.requires_grad = requires_grad
        else:
            self.src_copy_proj = lexicon

    def get_tgt_mask(self,T=None):
        if T is None:
            return self.tgt_mask
        else:
            return self.tgt_mask[:T,:T]

    def forward(self, x, cmd):
        S, B = cmd.shape
        # Language processing
        cmd_embd = self.inp_embed(cmd) + self.inp_pos_embed[:S, :, :]  # S X B X D
        cmd_embd = self.drop(cmd_embd)
        mask = (cmd.transpose(0,1) == self.vocab.pad())  # B X S
        source = self.encoder(cmd_embd, src_key_padding_mask=mask)  # S X B X D
        # Image processing
        _, encodings, *_ = self.vqvae.encode(x)  # B x H x W
        enc_T = encodings.view(B, -1).transpose(0, 1)  # L X B (L = HW)
        L, _ = enc_T.shape
        dec_inp_ids = enc_T[:-1, :]
        posx_embed = self.out_posx_embed[np.arange(L-1) % self.vqvae.latent_shape[1]]
        posy_embed = self.out_posy_embed[np.arange(L-1) // self.vqvae.latent_shape[1]]
        out_pos_embed = torch.cat((posx_embed, posy_embed), dim=-1)
        out_embed = self.out_embed(dec_inp_ids) + out_pos_embed
        out_embed = self.drop(out_embed)
        start_embed = self.start_embed.expand(1, B, self.rnn_dim)
        dec_inp_embeds = torch.cat((start_embed, out_embed), dim=0)
        # Decode
        # pdb.set_trace()
        tgt_mask = self.get_tgt_mask()
        decoded, src_att_weights, self_att_weights = self.decoder(dec_inp_embeds, source, memory_key_padding_mask=mask, tgt_mask=tgt_mask)
        # decoded =  self.decoder(dec_inp_embeds, source, memory_key_padding_mask=mask, tgt_mask=tgt_mask)
        # decoded: L x B x D
        # ## DO COPY
        src_att_probs = F.softmax(src_att_weights, dim=-1)  # B x L x S FIXME: I patched torch package to remove dropout from returned attweights
        src_proj = self.src_copy_proj(cmd).transpose(0, 1)  # S x B x N -> B x S x N
        src_translate_prob = src_att_probs @ src_proj  # B x L x N
        src_translate_prob = src_translate_prob.transpose(0, 1)  # L x B x N
        # self_att_weights = self.drop(self_att_weights[:,:,1:]) + EPS
        # self_att_probs = F.softmax(torch.log(self_att_weights),dim=-1)
        # self_proj = self.self_copy_proj(dec_inp_ids).transpose(0,1)
        # self_translate_prob = self_att_probs @ self_proj
        # self_translate_prob = self_translate_prob.transpose(0,1)
        # Seqs
        seq_logits = self.seq_picker(decoded)
        seq_weight = torch.exp(seq_logits)
        # seq_copy    = torch.sigmoid(seq_logits) # L X B X 3
        # pdb.set_trace()
        # seq_nll = -F.log_softmax(seq_logits,dim=-1).view(-1,seq_logits.size(-1))
        # seq_weights = F.softmax(seq_logits,dim=-1).unsqueeze(2) + EPS # L x B x 2 -> L x B x 1 x 2
        output = enc_T.flatten()
        src_copy_probs = src_translate_prob.contiguous().view(-1,src_translate_prob.size(-1))
        must_copied = src_copy_probs[range(src_copy_probs.size(0)), output] > 0.05 # 0.3must_
        loss1 = self.copy_criterion(seq_logits.flatten(),must_copied.float())
        ###
        # scores: prob_logits, prob_copy_logits, prob_self_copy_logits
        # probs: prob_write, prob_copy, prob_self_copy # L X B X N (number of  codes)
        # final_probs: (alpha_1 * prob_write + alpha_2 * prob_copy...)
        #
        # multiprob = sigmoid(prob_copy)
        #
        #
        # prob_copy_logits = [[0, 0, 10], [10, 10, 0]]
        # alpha = [100, 1]
        # prob_logits = [[20, 10, 15], [20, 10, 15]]
        #
        # [[0, 0, 1], [softmax([0, 0, -inf] + prob_logits)]]
        ###
        # logits = F.softmax(self.proj(self.highdrop(decoded)),dim=-1)  # L x B x N
        logits = self.proj(self.highdrop(decoded))  # L x B x N
        # copy_logits = torch.stack((src_translate_prob,self_translate_prob),3) # L x B x N x 2

        logits = seq_weight * torch.log(src_translate_prob + EPS) + F.log_softmax(logits,dim=-1)
        # logits  = self.proj(decoded)  # L x B x N
        ###
        # outs = torch.stack((logits,src_translate_prob,self_translate_prob),3) # L x B x N x 2
        # logits = torch.log((seq_weights * outs).sum(-1) + EPS)  # L x B x N x 2 -> # L x B x N
        # Loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), output)
        return loss + loss1

    @torch.no_grad()
    def predict(self, cmd, top_k=None, sample=True):
        S, B = cmd.shape
        C, H, W = self.vqvae.latent_shape

        cmd_embd = self.inp_embed(cmd) + self.inp_pos_embed[:S, :, :]
        mask = (cmd.transpose(0, 1) == self.vocab.pad())
        source = self.encoder(cmd_embd, src_key_padding_mask=mask)

        out_embed = torch.zeros(H*W, B, self.rnn_dim, device=source.device)
        next_embed = self.start_embed
        encodings = []
        copy_probs = []
        src_proj = self.src_copy_proj(cmd).transpose(0,1)
        for i in range(self.outlen):
            out_embed[i] = next_embed
            tgt_mask = self.get_tgt_mask(i+1)
            decoded, src_att_weights, self_att_weights = self.decoder(out_embed[:i+1], source, memory_key_padding_mask=mask, tgt_mask=tgt_mask)
            #decoded =  self.decoder(out_embed[:i+1], source, memory_key_padding_mask=mask, tgt_mask=tgt_mask)
            src_att_probs = F.softmax(src_att_weights[:, -1:, :], dim=-1)  # B x 1 x S
            src_translate_prob = src_att_probs @ src_proj # B x 1 x N
            src_translate_prob = src_translate_prob.squeeze(1)  # B x N
            # if i > 0:
            #     self_att_probs = F.softmax(torch.log(self_att_weights[:,-1:,1:]+EPS), dim=-1) # B x 1 x L
            #     self_proj = self.self_copy_proj(torch.stack(encodings,dim=1))
            #     self_translate_prob = self_att_probs @ self_proj # B x 1 x N
            #     self_translate_prob = self_translate_prob.squeeze(1)  # B x N
            # else:
            #     self_translate_prob = F.softmax(src_translate_prob * 0, dim=-1) # uniform
            seq_logits = torch.exp(self.seq_picker(decoded[-1]))
            copy_probs.append(seq_logits)
            # pdb.set_trace()
            # seq_weights = F.softmax(seq_logits ,dim=-1) + EPS # B x 2
            logits = self.proj(decoded[-1])  # B X N
            # outs = torch.stack((logits,src_translate_prob,self_translate_prob),2)
            # logits = torch.log((seq_weights.unsqueeze(1) * outs).sum(-1) + EPS)
            logits = seq_logits * torch.log(src_translate_prob + EPS) + F.log_softmax(logits,dim=-1)
            # logits  = F.log_softmax(logits,dim=-1)
            # pdb.set_trace()
            # logits = self.proj(decoded[-1])
            if top_k is not None:
                logits = top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            # pdb.set_trace()
            if sample:
                idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                idx = torch.argmax(probs, dim=-1)
            encodings.append(idx)
            posx_embed = self.out_posx_embed[i %  self.vqvae.latent_shape[1]]
            posy_embed = self.out_posy_embed[i //  self.vqvae.latent_shape[1]]
            out_pos_embed = torch.cat((posx_embed,posy_embed),dim=-1)
            next_embed = (self.out_embed(idx) + out_pos_embed).unsqueeze(0)

        encodings = torch.stack(encodings,dim=1)
        quantized = self.vqvae.codebook1._embedding(encodings) # B,HW,C
        z_rnn = quantized.transpose(1,2).contiguous().view(B, C, H, W)
        x_tilde = self.vqvae.decode(z_rnn)
        copy_probs = torch.stack(copy_probs, dim=1).view(B, H, W)
        return x_tilde, encodings.view(B,H,W), copy_probs

    def make_number_grid(self, encodings):
        B = encodings.shape[0]
        return torch.stack([self.number_matrix(encodings[i]) for i in range(B)],dim=0)

    def number_matrix(self, encoding, size=(512,512)):
        # Open image file
        h, w = encoding.shape
        encoding = encoding.numpy()
        # Set up figure
        my_dpi = 300
        fig = plt.figure(figsize=(size[0] / my_dpi, size[1] / my_dpi), dpi=my_dpi)
        ax = fig.add_subplot(111)

        # Remove whitespace from around the image
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

        # Set the gridding interval: here we use the major tick interval
        myInterval = 1.0 / h
        loc = plticker.MultipleLocator(base=myInterval)
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)

        # Add the grid
        ax.grid(which='major', axis='both', linestyle='-', linewidth=0.2)

        # Find number of gridsquares in x and y direction
        nx = h
        ny = w
        colors = ['yellow', 'blue', 'gray', 'brown', 'cyan', 'purple', 'red', 'green']
        # Add some labels to the gridsquares
        for j in range(ny):
            y = myInterval/2 + j*myInterval
            for i in range(nx):
                x = myInterval/2. + i*myInterval
                code = encoding[j, i]
                color = 'black'
                if code in self.rev_matchings:
                    if self.rev_matchings[code] in colors:
                        color = self.rev_matchings[code]
                ax.text(x, y, str(encoding[j, i]), ha='center', va='center', fontsize=4.5, color=color)

        # Save the figure
        figtensor = fig2tensor(fig)
        plt.cla()
        plt.close(fig)
        return figtensor

    def make_copy_grid(self, encodings):
        B = encodings.shape[0]
        return torch.stack([self.copy_heatmap(encodings[i]) for i in range(B)], dim=0)

    def copy_heatmap(self, copy_probs, size=(512,512)):
        # Open image file
        h, w = copy_probs.shape
        copy_probs = copy_probs.numpy()
        # Set up figure
        my_dpi = 300
        fig = plt.figure(figsize=(size[0] / my_dpi, size[1] / my_dpi), dpi=my_dpi)
        ax = fig.add_subplot(111)
        # Remove whitespace from around the image
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.imshow(copy_probs, cmap='hot', interpolation='nearest')
        figtensor = fig2tensor(fig)
        plt.cla()
        plt.close(fig)
        return figtensor
