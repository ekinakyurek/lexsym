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
from src import utils
from src import parallel
from src.vqa import VQA
from src.datasets import get_data
from seq2seq.src import NoamLR

import torchvision
from torchvision.utils import make_grid


FLAGS = flags.FLAGS

flags.DEFINE_string("modeltype", default='VQA',
                    help='VQA')

flags.DEFINE_string('vae_path', default='',
                    help='A pretrained vae path for conditional vae models.')

flags.DEFINE_string("lex_path", default='',
                    help='A prelearned lexicon path to be used in text-image '
                         'vqvae models')

flags.DEFINE_string("code_files", default='',
                    help='Pre cached codes for images')

flags.DEFINE_integer("warmup_steps", default=-1,
                     help="noam warmup_steps")


def evaluate_model(model, test_loader, code_cache=None, gpu=None):
    val_nll = 0.
    correct = 0.
    total = 0.
    for (question, img, answer, files) in iter(test_loader):
        question = question.transpose(0, 1)
        if code_cache is not None:
            img = torch.stack([code_cache[f] for f in files], dim=0)
        if gpu is not None:
            img = img.cuda(gpu, non_blocking=True)
            question = question.cuda(gpu, non_blocking=True)
        if torch.cuda.is_available():
            answer = answer.cuda(gpu, non_blocking=True)

        nll, pred = model(**dict(question=question, img=img, answer=answer, predict=True))
        correct += (pred == answer).sum().item()
        total += len(answer)
        val_nll += nll.mean().item()

        for i in range(img.shape[0]):
            q_str = " ".join(test_loader.dataset.vocab.decode(question[i].cpu().numpy()))
            a_str = test_loader.dataset.answer_vocab._rev_contents[answer[i].item()]
            f_str = files[i]
            p_str = test_loader.dataset.answer_vocab._rev_contents[pred[i].item()]
            logging.info(f"{i}\t{q_str}\t{a_str}\t{p_str}\t{f_str}")

        if total > 1000:
            break
    return val_nll / total, correct / total


def train_vqa_model_model(model,
                          train,
                          test,
                          vis_folder,
                          warmup_steps=-1,
                          lexicon=None,
                          code_cache=None,
                          optimizer=None,
                          start_iter=0,
                          visualize_every=1000,
                          n_batch=64,
                          n_iter=1,
                          n_workers=1,
                          distributed=False,
                          ngpus_per_node=1,
                          gpu=0,
                          rank=0,
                          lr=0.0001):

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    if warmup_steps != -1:
        scheduler = NoamLR(optimizer, model.module.rnn_dim, warmup_steps=warmup_steps)
        print('using noam scheduler')
    else:
        scheduler = None

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
                             batch_size=n_batch,
                             shuffle=True,
                             collate_fn=train.collate,
                             num_workers=n_workers,
                             worker_init_fn=utils.worker_init_fn)

    # writer = utils.get_tensorboard_writer()
    total, nll, epoch = [0.] * 3
    train_iter = iter(train_loader)
    model.train()
    for i in tqdm(range(start_iter, n_iter)):
        try:
            question, img, answer, files = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            epoch += 1
            if train_sampler:
                train_sampler.set_epoch(epoch)  # FIX
            question, img, answer, files = next(train_iter)

        question = question.transpose(0, 1)

        if code_cache is not None:
            img = torch.stack([code_cache[f] for f in files], dim=0)
            if lexicon is not None:
                utils.random_swap(lexicon,
                                  question,
                                  train.vocab,
                                  answer,
                                  train.answer_vocab,
                                  img)
        if gpu is not None:
            img = img.cuda(gpu, non_blocking=True)
            question = question.cuda(gpu, non_blocking=True)
            answer = answer.cuda(gpu, non_blocking=True)

        loss = model(**dict(question=question, img=img, answer=answer))
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total += img.shape[0]
        nll += loss.item()

        if main_worker and (i+1) % visualize_every == 0:
            with torch.no_grad():
                logging.info('%d iterations', (i+1))
                logging.info('Train Loss: %.6f', (nll / total))
                model.eval()
                val_nll, val_acc = evaluate_model(model, test_loader, code_cache, gpu=gpu)
                print(f"val_nll: {val_nll} val_acc: {val_acc}")
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'iter': i,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, filename=os.path.join(vis_folder,
                                         'checkpoint.pth.tar'))
                model.train()


def train_vqa_model(gpu, ngpus_per_node, args):
    args.gpu = gpu
    args.ngpus_per_node = ngpus_per_node
    parallel.init_distributed(args)

    if args.code_files:
        code_cache = {}
        files = args.code_files.split(',')
        for file in files:
            with open(file) as handle:
                for line in handle:
                    _, code, filename = line.split('\t')
                    code_cache[filename.strip()] = torch.tensor(list(map(int, code.split())))
    else:
        code_cache = None

    if args.lex_path:
        with open(args.lex_path) as handle:
            lexicon = json.load(handle)
        lexicon = utils.filter_lexicon(lexicon)
        print(lexicon)
    else:
        lexicon = None

    train, test = get_data(vqa=True, no_images=code_cache is not None)
    vis_folder = utils.flags_to_path()
    os.makedirs(vis_folder, exist_ok=True)
    logging.info("vis folder: %s", vis_folder)

    if args.modeltype == "VQA":
        model = VQA(3,
                    args.h_dim,
                    args.n_latent,
                    train.vocab,
                    train.answer_vocab,
                    rnn_dim=args.rnn_dim,
                    n_codes=args.n_codes,
                    cc=args.commitment_cost,
                    decay=args.decay,
                    epsilon=args.epsilon,
                    beta=args.beta,
                    cmdproc=False,
                    size=train.size)
    else:
        raise ValueError(f"Not supported model type {args.modeltype}")

    if args.resume == '':
        args.resume = args.vae_path
        utils.resume(model.vqvae, args, mark='iter')
        optimizer = None
        args.start_iter = 0
    else:
        args.start_iter = 0
        optimizer = utils.resume(model, args, mark='iter')

    model = parallel.distribute(model, args)

    train_vqa_model_model(model,
                          train,
                          test,
                          vis_folder,
                          lexicon=lexicon,
                          code_cache=code_cache,
                          optimizer=optimizer,
                          warmup_steps=args.warmup_steps,
                          start_iter=args.start_iter,
                          visualize_every=args.visualize_every,
                          n_batch=args.n_batch,
                          n_iter=args.n_iter,
                          n_workers=args.n_workers,
                          distributed=args.distributed,
                          ngpus_per_node=ngpus_per_node,
                          gpu=args.gpu,
                          rank=args.rank,
                          lr=args.lr)
    

    if args.distributed:
        utils.cleanup()


def main(_):
    parallel.run_parallel(train_vqa_model)


if __name__ == "__main__":
    app.run(main)
