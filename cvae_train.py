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
from src import utils
from src.lex import FilterModel
from src.vqvae import VectorQuantizedVAE
from src.cvqvae import CVQVAE
from src.cvae import CVAE
from src.dae import DAE
from src import parallel
from src.datasets import get_data

import torchvision
from torchvision.utils import make_grid
from torchvision.utils import save_image

FLAGS = flags.FLAGS


flags.DEFINE_string("lex_path", default='',
                    help='A prelearned lexicon path to be used in text-image '
                         'vqvae models')

flags.DEFINE_string('vae_path', default='',
                    help='A pretrained vae path for conditional vae models.')


def evaluate_cvae(model,loader):
    val_recon_error = 0.0
    val_loss = 0.0
    cnt = 0
    for (cmd, img, _) in loader:
        img = img.to(utils.device)
        cmd = cmd.to(utils.device)
        cnt += img.shape[0]
        loss = model(img, cmd)
        x_tilde, *_ = model.predict(cmd)
        val_recon_error += (x_tilde-img).pow(2).sum().item()
        val_loss += loss.item() * img.shape[0]
        if cnt > 100:
            break
    return val_recon_error / cnt, val_loss / cnt


def train_cvae():
    assert FLAGS.vae_path != ""

    train, test = get_data()
    vis_folder = utils.flags_to_path()
    os.makedirs(vis_folder, exist_ok=True)
    logging.info("vis folder: %s", vis_folder)

    if FLAGS.test:
        model = torch.load(FLAGS.model_path)
        model = model.to(utils.device)

        loader = DataLoader(train,
                            batch_size=FLAGS.n_batch,
                            shuffle=True,
                            collate_fn=train.collate,
                            num_workers=FLAGS.n_workers)

        test_loader = DataLoader(test,
                                 batch_size=36,
                                 shuffle=True,
                                 collate_fn=train.collate,
                                 num_workers=FLAGS.n_workers)
    else:
        if FLAGS.modeltype == "CVQVAE":
            model = CVQVAE(3,
                           FLAGS.h_dim,
                           FLAGS.n_latent,
                           train.vocab,
                           rnn_dim=FLAGS.rnn_dim,
                           n_codes=FLAGS.n_codes,
                           cc=FLAGS.commitment_cost,
                           decay=FLAGS.decay,
                           epsilon=FLAGS.epsilon,
                           beta=FLAGS.beta,
                           cmdproc=False,
                           size=train.size,
                           ).to(utils.device)
            model.vqvae.load_state_dict(torch.load(FLAGS.vae_path).state_dict())
            for p in model.vqvae.parameters():
                p.requires_grad = False
            if FLAGS.lex_path != "":
                model.set_src_copy(model.get_src_lexicon(FLAGS.lex_path))

            model = model.to(utils.device)

            loader = DataLoader(train,
                                batch_size=FLAGS.n_batch,
                                shuffle=True, collate_fn=train.collate,
                                num_workers=FLAGS.n_workers)
            test_loader = DataLoader(test,
                                     batch_size=32,
                                     shuffle=True,
                                     collate_fn=train.collate,
                                     num_workers=FLAGS.n_workers)

            model.eval()
            T = torchvision.transforms.ToPILImage(mode=train.color)
            with torch.no_grad():
                sample, _ = model.vqvae.sample(B=32)
                sample = sample.cpu().data * train.std[None, :, None, None] + train.mean[None, :, None, None]
                T(make_grid(sample.clip_(0, 1))).convert("RGB").save(os.path.join(vis_folder, "initial_samples.png"))
                generator = iter(test_loader)
                cmd, img, _ = next(generator)
                cmd = cmd.to(utils.device)
                img = img.to(utils.device)
                _, recon, *_ = model.vqvae(img, cmd)
                recon = recon.cpu().data * train.std[None, :, None, None] + train.mean[None, :, None, None]
                img = img.cpu().data * train.std[None, :, None, None] + train.mean[None, :, None, None]
                res = torch.cat((recon, img), 0).clip_(0, 1)
                T(make_grid(res)).convert("RGB").save(os.path.join(vis_folder, f"initial_recons.png"))

        elif FLAGS.modeltype == "CVAE":
            model = CVAE(3,
                         FLAGS.h_dim,
                         FLAGS.n_latent,
                         train.vocab,
                         beta=FLAGS.beta,
                         rnn_dim=FLAGS.rnn_dim,
                         size=train.size)

            model.vae.load_state_dict(torch.load(FLAGS.vae_path).state_dict())
            model = model.to(utils.device)
            for p in model.vae.parameters():
                p.requires_grad = False
        else:
            raise ValueError(f"Unknown model type {FLAGS.modeltype}")

        optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
        train_loss = 0.
        cnt = 0.

        generator = iter(loader)
        model.train()
        model.vqvae.eval()
        for i in tqdm(range(FLAGS.n_iter)):
            try:
                cmd, img, _ = next(generator)
            except StopIteration:
                generator = iter(loader)
                cmd, img, _ = next(generator)

            cmd = cmd.to(utils.device)
            img = img.to(utils.device)

            optimizer.zero_grad()
            loss = model(img, cmd)
            loss.backward()
            optimizer.step()

            train_loss += loss * img.shape[0]
            cnt += img.shape[0]

            if i == 10 or (i+1) % 100 == 0:
                model.eval()
                with torch.no_grad():
                    logging.info('%d iterations', (i+1))
                    logging.info('%.6f train loss', (train_loss / cnt))
                    val_recon, val_loss = evaluate_cvae(model, test_loader)
                    logging.info('val_recon_error: %.6f', val_recon)
                    logging.info('val_loss: %.6f', val_loss)
                    T = torchvision.transforms.ToPILImage(mode=train.color)
                    test_iter = iter(test_loader)
                    for j in range(5):
                        cmd, img, _ = next(test_iter)
                        cmd = cmd.to(utils.device)
                        img = img.to(utils.device)
                        recon, pred_encodings, copy_probs = model.predict(cmd, sample=True, top_k=10)
                        pred_encodings = model.make_number_grid(pred_encodings.cpu())
                        copy_heatmap = model.make_copy_grid(copy_probs.cpu())
                        _, encodings, *_ = model.vqvae.encode(img)
                        encodings = model.make_number_grid(encodings.cpu())
                        encodings = torch.cat((copy_heatmap, pred_encodings, encodings), 0)
                        recon = recon.cpu().data * train.std[None, :, None, None] + train.mean[None, :, None, None]
                        img = img.cpu().data * train.std[None, :, None, None] + train.mean[None, :, None, None]
                        res = torch.cat((recon, img), 0).clip_(0, 1)
                        T(make_grid(encodings, nrow=encodings.shape[0]//3)).convert("RGB").save(os.path.join(vis_folder, f"{i}_{j}_encodings.png"))
                        T(make_grid(res, nrow=res.shape[0]//2)).convert("RGB").save(os.path.join(vis_folder, f"{i}_{j}.png"))
                        logging.info("saved")
                model.train()
                model.vqvae.eval()
        torch.save(model, os.path.join(vis_folder, f"model.pt"))

    model.eval()
    with torch.no_grad():
        val_recon, val_loss = evaluate_cvae(model, test_loader)
        logging.info('val_recon_error: %.6f', val_recon)
        logging.info('val_loss: %.6f', val_loss)
        T = torchvision.transforms.ToPILImage(mode=train.color)
        test_iter = iter(test_loader)
        for j in range(5):
            cmd, img, _ = next(test_iter)
            cmd = cmd.to(utils.device)
            img = img.to(utils.device)
            logging.info("sampling")
            recon, *_ = model.predict(cmd, top_k=10, sample=True)
            logging.info("sampled")
            recon = recon.cpu().data * train.std[None, :, None, None] + train.mean[None, :, None, None]
            img = img.cpu().data * train.std[None, :, None, None] + train.mean[None, :, None, None]
            res = torch.cat((recon, img), 0).clip_(0, 1)
            logging.info("saving samples")
            T(make_grid(res, nrow=res.shape[0]//2)).convert("RGB").save(os.path.join(vis_folder, f"eval_{j}.png"))
            logging.info("saved")


def main(_):
    train_cvae()


if __name__ == "__main__":
    app.run(main)
