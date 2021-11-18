import os
import random
import json
import sys
import itertools
import functools
import warnings
import imageio
import shutil

import numpy as np
import torch

from absl import app, flags, logging
from tqdm import tqdm

from torch import optim
from torch.utils import data as torch_data
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

import options
from src.lex import FilterModel
from src.vqvae import VectorQuantizedVAE
from src.vae import VAE
from src.dae import DAE
from src import utils
from src import parallel
from src.datasets import get_data

import torchvision
from torchvision.utils import make_grid

FLAGS = flags.FLAGS


flags.DEFINE_string("modeltype", default='VQVAE',
                    help='VAE, VQVAE, TODO: fix this flag for filter model')


def evaluate_vqvae(model, test_loader, gpu=None):
    val_res_recon_error = 0.0
    val_res_nll = 0.0
    cnt = 0
    for (cmd, img, _) in iter(test_loader):
        if gpu is not None:
            img = img.cuda(gpu, non_blocking=True)
            cmd = cmd.cuda(gpu, non_blocking=True)
        cnt += img.shape[0]
        _, _, recon_error, *_ = model(**dict(img=img, cmd=cmd))
        nll = model.nll(**dict(img=img, cmd=cmd))
        val_res_recon_error += recon_error.item()
        val_res_nll += nll.item()

    return val_res_recon_error / cnt, val_res_nll / cnt

def visualize_vae(model, test_loader, train, vis_folder, i=0, gpu=None):
    T = torchvision.transforms.ToPILImage(mode=train.color)
    test_iter = iter(test_loader)
    for j in range(5):
        cmd, img, _ = next(test_iter)
        if gpu is not None:
            img = img.cuda(gpu, non_blocking=True)
            cmd = cmd.cuda(gpu, non_blocking=True)
        loss, recon, recon_error, *_ = model(**dict(img=img, cmd=cmd))
        recon = recon.cpu().data * train.std[None, :, None, None] + train.mean[None, :, None, None]
        img = img.cpu().data * train.std[None, :, None, None] + train.mean[None, :, None, None]
        res = torch.cat((recon, img), 0).clip_(0, 1)
        T(make_grid(res)).convert("RGB").save(os.path.join(vis_folder, f"{i}_{j}.png"))
        sample, *_ = model.sample(B=32)
        sample = sample.cpu().data * train.std[None, :, None, None] + train.mean[None, :, None, None]
        T(make_grid(sample.clip_(0, 1))).convert("RGB").save(os.path.join(vis_folder, f"prior_{i}_{j}.png"))

def train_vae_model(model,
                    train,
                    test,
                    vis_folder,
                    optimizer=None,
                    start_iter=0,
                    visualize_every=10,
                    n_batch=64,
                    n_iter=1,
                    n_workers=1,
                    distributed=False,
                    ngpus_per_node=1,
                    gpu=0,
                    rank=0,
                    kl_anneal=False,
                    decoder_reset=-1,
                    lr=0.0001):

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    main_worker = rank % ngpus_per_node == 0
    logging.info(ngpus_per_node)

    if distributed:
        train_sampler = DistributedSampler(train)
    else:
        train_sampler = None

    worker_init_fn = functools.partial(utils.worker_init_fn, rank=rank)

    train_loader = DataLoader(train,
                              batch_size=n_batch,
                              shuffle=(train_sampler is None),
                              collate_fn=train.collate,
                              sampler=train_sampler,
                              num_workers=n_workers,
                              worker_init_fn=worker_init_fn)

    test_loader = DataLoader(test,
                             batch_size=32,
                             shuffle=True,
                             collate_fn=train.collate,
                             num_workers=n_workers,
                             worker_init_fn=utils.worker_init_fn)

    # writer = utils.get_tensorboard_writer()
    train_res_recon_error = 0.
    cnt = 0.
    epoch_count = 0
    train_iter = iter(train_loader)
    model.train()
    for i in range(start_iter, n_iter):
        try:
            cmd, img, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            epoch_count += 1
            if train_sampler:
                train_sampler.set_epoch(epoch_count)  # FIX
            cmd, img, _ = next(train_iter)

        cmd = cmd.transpose(0, 1)
        if gpu is not None:
            cmd = cmd.cuda(gpu, non_blocking=True)
            img = img.cuda(gpu, non_blocking=True)

        loss, errors = model(**dict(img=img, cmd=cmd, only_loss=True))
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        train_res_recon_error += errors['reconstruction_error'].mean().item()
        cnt += img.shape[0]
        # train_res_nll.append(nll.item())

        if main_worker and (i+1) % visualize_every == 0:
            with torch.no_grad():
                logging.info('%d iterations', (i+1))
                logging.info('recon_error: %.6f',
                             (train_res_recon_error / cnt))
                model.eval()
                visualize_vae(model, test_loader, train, vis_folder, i=i, gpu=gpu)
                utils.save_checkpoint({
                    'epoch': epoch_count + 1,
                    'iter': i,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, filename=os.path.join(vis_folder,
                                         'checkpoint.pth.tar'))
                model.train()


def train_vae(gpu, ngpus_per_node, args):
    args.gpu = gpu
    args.ngpus_per_node = ngpus_per_node
    parallel.init_distributed(args)

    train, test = get_data()
    vis_folder = utils.flags_to_path()
    os.makedirs(vis_folder, exist_ok=True)
    logging.info("vis folder: %s", vis_folder)

    if args.modeltype == "VQVAE":
        model = VectorQuantizedVAE(3, args.h_dim,
                                   args.n_latent,
                                   n_codes=args.n_codes,
                                   cc=args.commitment_cost,
                                   decay=args.decay,
                                   epsilon=args.epsilon,
                                   beta=args.beta,
                                   cmdproc=False,
                                   size=train.size,
                                   )
    elif args.modeltype == "VAE":
        model = VAE(3,
                    args.h_dim,
                    args.n_latent,
                    beta=args.beta,
                    size=train.size)
    elif args.modeltype == "DAE":
        model = DAE(3,
                    latentdim=args.n_latent*4*4,
                    size=train.size)
    else:
        raise ValueError(f"Unknown model type {args.modeltype}")

    model = parallel.distribute(model, args)

    if not hasattr(model, 'sample'):
        model.sample = model.module.sample

    args.start_iter = 0

    optimizer = utils.resume(model, args, mark='iter')

    train_vae_model(model,
                    train,
                    test,
                    vis_folder,
                    optimizer=optimizer,
                    start_iter=args.start_iter,
                    visualize_every=args.visualize_every,
                    n_batch=args.n_batch,
                    n_iter=args.n_iter,
                    n_workers=args.n_workers,
                    distributed=args.distributed,
                    ngpus_per_node=ngpus_per_node,
                    gpu=args.gpu,
                    rank=args.rank,
                    lr=args.lr,
                    kl_anneal=args.kl_anneal,
                    decoder_reset=args.decoder_reset)

    if args.distributed:
        utils.cleanup()


def main(_):
    parallel.run_parallel(train_vae)


if __name__ == "__main__":
    app.run(main)
