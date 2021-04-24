import torch
from torch import nn, optim
import torch.utils.data as torch_data
import torch.nn.functional as F
import numpy as np
import random
import sys
from .src import batch_seqs
#import pdb
EPS = 1e-7

def encode(data, vocab_x, vocab_y):
    encoded = []
    for datum in data:
        encoded.append(encode_io(datum, vocab_x, vocab_y))
    return encoded

def encode_io(datum, vocab_x, vocab_y):
    inp, out = datum
    return (encode_input(inp),encode_output(out))

def encode_input(inp, vocab_x):
    return [vocab_x.sos()] + vocab_x.encode(inp) + [vocab_x.eos()]

def encode_output(out, vocab_y):
    return [vocab_y.sos()] + vocab_y.encode(out) + [vocab_y.eos()]


def eval_format(vocab, seq):
    if vocab.eos() in seq:
        seq = seq[:seq.index(vocab.eos())+1]
    seq = seq[1:-1]
    return vocab.decode(seq)

def collate(batch):
    batch = sorted(batch,
                   key=lambda x: len(x[0]),
                   reverse=True)
    inp, out = zip(*batch)
    lens = torch.LongTensor(list(map(len,inp)))
    inp = batch_seqs(inp)
    out = batch_seqs(out)
    return inp, out, lens
