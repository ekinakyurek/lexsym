import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import math
import io
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import torchvision.transforms.functional as TF
import os
from absl import flags
from absl import logging
from torch.utils.tensorboard import SummaryWriter

FLAGS = flags.FLAGS

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.normal_(m.weight.data, std=0.002)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


def reset_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.normal_(m.weight.data, std=0.002)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)
    elif classname.find('Linear') != -1:
        m.reset_parameters()
    elif classname.find('Embedding') != -1:
        m.reset_parameters()

class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape,  # extra comma

    def forward(self, x):
        return x.view(*self.shape)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, *x):
        return self.lambd(*x)


def conv3x3(input_dim, output_dim, kernel_dim=3):
    return Residual(nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_dim, padding="same"),
                nn.LeakyReLU(0.2)))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def top_k_logits(logits, k):
    v, _ = torch.topk(logits, k)
    return logits.masked_fill(logits < v[:, [-1]], -float('Inf'))

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            ix = torch.argmax(probs, dim=-1, keepdim=True)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x

def fig2tensor(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return TF.to_tensor(img)

def make_number_grid(encodings):
    B = encodings.shape[0]
    return torch.stack([number_matrix(encodings[i]) for i in range(B)],dim=0)

def number_matrix(encoding, size=(512,512)):
    # Open image file
    h, w = encoding.shape
    encoding = encoding.numpy()
    # Set up figure
    my_dpi = 300
    fig = plt.figure(figsize=(size[0] / my_dpi, size[1] / my_dpi),dpi=my_dpi)
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

    # Add some labels to the gridsquares
    for j in range(ny):
        y = myInterval/2 + j*myInterval
        for i in range(nx):
            x = myInterval/2. + i*myInterval
            ax.text(x, y, str(encoding[j,i]),ha='center',va='center', fontsize=4.5)

    # Save the figure
    figtensor = fig2tensor(fig)
    plt.cla()
    plt.close(fig)
    return figtensor


def cpu(tensors):
    if type(tensors) == list:
        return [tensor.detach().cpu() for tensor in tensors]
    else:
        return tensors.detach().cpu()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def init_process(rank=0, size=1, backend='nccl', init_method="tcp://127.0.0.1:23456"):
    """ Initialize the distributed environment. """
    torch.distributed.init_process_group(backend,
                                         rank=rank,
                                         world_size=size,
                                         init_method=init_method)


def cleanup():
    torch.distributed.destroy_process_group()


def worker_init_fn(worker_id, rank=0):
    np.random.seed(np.random.get_state()[1][0] + worker_id + rank)


class ConfigDict(object):
    def __init__(self, my_dict):
        self._initial_dict = my_dict
        for key in my_dict:
            setattr(self, key, my_dict[key])


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def flags_to_path():
    root = os.path.join(FLAGS.vis_root, FLAGS.datatype, FLAGS.modeltype)

    if "VQVAE" in FLAGS.modeltype:
        path = os.path.join(root,
                            (f"beta_{FLAGS.beta}_ncodes_{FLAGS.n_codes}_"
                             f"ldim_{FLAGS.n_latent}_dim_{FLAGS.h_dim}_"
                             f"lr_{FLAGS.lr}")
                            )
    elif FLAGS.modeltype == 'FilterModel':
        path = os.path.join(root, FLAGS.lex_vae_type,
                            (f"dim_{FLAGS.n_latent}_"
                             f"lr_{FLAGS.lr}_"
                             f"beta_{FLAGS.beta}")
                            )
    else:
        path = os.path.join(root,
                            (f"beta_{FLAGS.beta}_ldim_{FLAGS.n_latent}_"
                             f"dim_{FLAGS.h_dim}"
                             f"_lr_{FLAGS.lr}")
                            )
    return path


def get_tensorboard_writer():
    path = flags_to_path()
    if not hasattr(logging, 'tb_writer'):
        logging.tb_writer = SummaryWriter(path)
    return logging.tb_writer


def set_logging_format(format='%(asctime)s [%(filename)s:%(lineno)d] %(message)s'):
    logging._absl_handler.setFormatter(
        logging.logging.Formatter(format, "%H:%M:%S"))


def flags_to_args():
    flags_dict = {k: v.value for k, v in FLAGS.__flags.items()}
    return ConfigDict(flags_dict)


def resume(model, args, mark='epoch'):
    optimizer = None
    if args.resume != '':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            if not isinstance(checkpoint, dict):
                checkpoint = {'state_dict': checkpoint.state_dict()}
            setattr(args, f'start_{mark}', checkpoint.get(mark, 0))
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except RuntimeError:
                model.module.load_state_dict(checkpoint['state_dict'])

            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint.get(mark, 0)))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        print("Initialized the model from scratch")
    return optimizer


def filter_lexicon(lexicon):
    # deleted_keys = set()
    # for (k1, v1) in lexicon.items():
    #     if len(v1) > 2:
    #         deleted_keys.add(k1)
    #
    # for k in deleted_keys:
    #     del lexicon[k]
    #
    # deleted_keys = set()
    # for (k1, v1) in lexicon.items():
    #     for ci, count in v1.items():
    #         for (k2, v2) in lexicon.items():
    #             if k2 == k1:
    #                 continue
    #             if ci in v2:
    #                 deleted_keys.add(k1)
    #                 deleted_keys.add(k2)
    # for k in deleted_keys:
    #     del lexicon[k]
    #
    # lexicon.pop('is')
    # lexicon.pop('as')
    # lexicon.pop('material')
    # lexicon.pop('spher')
    # lexicon.pop('10')
    keys_to_hold = "yellow,red,green,cyan,purple,blue,gray,brown".split(",")
    deleted_keys = set()
    for k in lexicon.keys():
        if k not in keys_to_hold:
            deleted_keys.add(k)

    for k in deleted_keys:
        del lexicon[k]

    return lexicon


def swap_ids(tensor, id1, id2):
    tensor.masked_fill_(tensor == id1, -1)
    tensor.masked_fill_(tensor == id2, id1)
    tensor.masked_fill_(tensor == -1, id2)


def random_swap(lexicon, question, vocab, answer, answer_vocab, codes):
    keys = filter(lambda k: vocab[k] in question or vocab[k] in answer, lexicon.keys())
    keys = list(keys)
    if len(keys) == 0:
        ks = random.sample(list(lexicon.keys()))
    else:
        k1 = random.choice(keys)
        all_keys = set(lexicon.keys())
        all_keys.remove(k1)
        k2 = random.choice(list(all_keys))
        ks = [k1, k2]

    ks_q_id = [vocab[k] for k in ks]
    ks_a_id = [answer_vocab[k] for k in ks]

    swap_ids(question, *ks_q_id)
    swap_ids(answer, *ks_a_id)

    for v, _ in lexicon[ks[0]].items():
        codes.masked_fill_(codes == int(v), -1)

    for v, _ in lexicon[ks[1]].items():
        code1 = random.choice(list(lexicon[ks[0]].keys()))
        codes.masked_fill_(codes == int(v), int(code1))

    code2 = random.choice(list(lexicon[ks[1]].keys()))

    codes.masked_fill_(codes == -1, int(code2))
