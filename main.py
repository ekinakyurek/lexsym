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

from src.lex import FilterModel
from src.vqvae import VectorQuantizedVAE
from src.vqvae import CVQVAE
from src.vae import VAE, CVAE
from src.dae import DAE
from src import utils

from data.shapes import ShapeDataset
from data.set import SetDataset
from data.scan import SCANDataset
from data.clevr import CLEVRDataset
import torchvision
from torchvision.utils import make_grid
from torchvision.utils import save_image

FLAGS = flags.FLAGS

flags.DEFINE_integer('h_dim', default=32,
                     help='Hidden dim in various models.')

flags.DEFINE_integer('rnn_dim', default=256,
                     help='RNN hidden dim.')

flags.DEFINE_integer('rnn_n_layers', default=2,
                     help='Number of layers for RNNs.')

flags.DEFINE_float('rnn_drop', default=0.1,
                   help='Dropout rate in RNNs.')

flags.DEFINE_integer('n_latent', default=24,
                     help='Latent dimension for vaes.')

flags.DEFINE_integer('lex_n_latent', default=64,
                     help="Latent dimension for lexical model's filters")

flags.DEFINE_integer('lex_n_steps', default=10,
                     help="Number of steps in lexical model")

flags.DEFINE_integer('lex_n_downsample', default=2,
                     help="Number of downsampling layers in lexical model")

flags.DEFINE_float('lex_att_loss_weight', default=0.0001,
                   help="Attention loss weight in lexical model")

flags.DEFINE_string('lex_vae_type', default='VAE',
                    help="VAE type for lexical model among VAE, VQVAE, None")

flags.DEFINE_bool('lex_text_conditional', default=False,
                  help="Determines VAE or CVAE for lexical model")

flags.DEFINE_bool('lex_append_z', default=False,
                  help="Appends z to encodings and embeddings")

flags.DEFINE_integer('n_batch', default=128,
                     help='Minibatch size to train.')

flags.DEFINE_integer('visualize_every', default=10,
                     help='Frequency of visualization.')

flags.DEFINE_integer('n_iter', default=100000,
                     help='Number of iteration to train. Might not be used if '
                          'n_epoch is used.')

flags.DEFINE_integer('n_epoch', default=50,
                     help='Number of epochs to train. Might not be used if '
                          'n_iter is used.')

flags.DEFINE_integer('n_codes', default=10,
                     help='Sets number of codes in vqvae.')

flags.DEFINE_integer('n_workers', default=4,
                     help='Sets num workers for data loaders.')

flags.DEFINE_integer('seed', default=0,
                     help='Sets global seed.')

flags.DEFINE_float('beta', default=1.0,
                   help='Sets beta parameter in beta vae.')

flags.DEFINE_float('commitment_cost', default=0.25,
                   help='Sets commitment lost in vqvae')

flags.DEFINE_string('datatype', default='setpp',
                    help='Sets which dataset to use.')

flags.DEFINE_string("modeltype", default='VQVAE',
                    help='VAE, VQVAE, TODO: fix this flag for filter model')

flags.DEFINE_string("vis_root", default='vis',
                    help='root folder for visualization and logs.')

flags.DEFINE_float('decay', default=0.99,
                   help='set learning rate value for optimizers')

flags.DEFINE_float('lr', default=1e-3,
                   help='Set learning rate for optimizers.')

flags.DEFINE_float('epsilon', default=1e-5,
                   help='Sets epsilon value in VQVAE.')

flags.DEFINE_bool("debug", default=False,
                  help='Enables debug mode.')

flags.DEFINE_bool('highdrop', default=False,
                  help='Enables high dropout to encourage copy.')

flags.DEFINE_bool('highdroptest', default=False,
                  help='Applies high dropout in test as well.')

flags.DEFINE_float("highdropvalue", default=0.,
                   help='High dropout value to encourage copying.')

flags.DEFINE_bool('copy', default=False,
                  help='Enable copy in seq2seq models')

flags.DEFINE_string('vae_path', default='',
                    help='A pretrained vae path for conditional vae models.')

flags.DEFINE_string("lex_path", default='',
                    help='A prelearned lexicon path to be used in text-image '
                         'vqvae models')

flags.DEFINE_string('model_path', default='',
                    help="Model path to load a pretrained model")

flags.DEFINE_bool('extract_codes', default=False,
                  help='Extract VQVAE codes for training and test set given a '
                       'pretrained vae')

flags.DEFINE_bool('filter_model', default=False,
                  help='To run filter model experiments.')

flags.DEFINE_bool('test', default=False,
                  help='Only runs evaluations.')


flags.DEFINE_bool('distributed', default=False,
                  help='Enables distributed data parallel.')

flags.DEFINE_integer('gpu', default=None,
                     help='Specifies which GPU to use. If None DataParallel '
                          'mode will be enabled')

flags.DEFINE_integer('rank', default=0,
                     help='Node rank for distributed training.')

flags.DEFINE_integer('world_size', default=1,
                     help='ngpus in distributed data parallel.')

flags.DEFINE_string('dist_backend', default='nccl',
                    help='Backend to use for distributed data parallel.')

flags.DEFINE_string('dist_url', default='tcp://127.0.0.1:23456',
                    help='Url used to set up distributed training.')

flags.DEFINE_bool('multiprocessing_distributed', False,
                  help='Use multi-processing distributed training to launch '
                       'N processes per node, which has N GPUs. This is the '
                       'fastest way to use PyTorch for either single node or '
                       'multi node data parallel training')

flags.DEFINE_string('tensorboard', default=None,
                    help='Use tensorboard for logging losses.')

flags.DEFINE_bool('kl_anneal', default=False,
                  help='Enables kl annealing.')

flags.DEFINE_integer('decoder_reset', default=-1,
                     help='Enables decoder reset for vae to prevent posterior collapse.')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def train_filter_model(model,
                       train,
                       test,
                       vis_folder,
                       visualize_every=10,
                       n_batch=64,
                       epoch=1,
                       n_workers=1,
                       distributed=False,
                       ngpus_per_node=1,
                       gpu=0,
                       rank=0,
                       kl_anneal=False,
                       decoder_reset=-1,
                       lr=0.0001,
                       ):

    optimizer = optim.Adam(model.parameters(), lr=lr)

    main_worker = rank % ngpus_per_node == 0
    logging.info(ngpus_per_node)

    if distributed:
        train_sampler = DistributedSampler(train)
    else:
        train_sampler = None

    worker_init_fn = functools.partial(utils.worker_init_fn, rank=rank)

    loader = DataLoader(train,
                        batch_size=n_batch,
                        shuffle=(train_sampler is None),
                        pin_memory=True,
                        collate_fn=train.collate,
                        sampler=train_sampler,
                        num_workers=n_workers,
                        worker_init_fn=worker_init_fn)

    total_steps = 0

    if hasattr(model, 'module'):
        model_object = model.module
    else:
        model_oject = model

    if model_object.vae is not None:
        variational = True
        if kl_anneal:
            target_beta = model_object.vae.beta
            model_object.vae.beta = 0.0
            kl_rate = 4*(target_beta / (epoch))
            logging.info(f"kl rate: {kl_rate}")
        if decoder_reset != -1:
            variational = False

    for i in range(epoch):

        if train_sampler:
            train_sampler.set_epoch(i)

        total_loss, total_item = .0, .0

        dloader = tqdm(loader) if main_worker else loader

        model.train()

        for (cmd, img, _) in dloader:
            cmd = cmd.transpose(0, 1)

            if gpu is not None:
                cmd = cmd.cuda(gpu, non_blocking=True)
                img = img.cuda(gpu, non_blocking=True)

            loss, scalars = model(cmd,
                                  img,
                                  test=False,
                                  variational=variational)
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_steps += 1
            total_loss += loss.item()
            total_item += img.shape[0]

            if main_worker:
                dloader.set_description(
                                 f"Epoch {i}, Avg Loss (Train): %.4f, N: %d" %
                                 (total_loss/total_item, total_item))

                if hasattr(logging, 'tb_writer'):
                    scalars = dict((k, v.mean().item())
                                   for (k, v) in scalars.items())
                    logging.tb_writer.add_scalars('Train/Loss', scalars,
                                                  total_steps)
                    logging.tb_writer.add_scalar('beta', model_object.vae.beta,
                                                 total_steps)

        if not distributed and not variational and i == decoder_reset:
            model = model_object
            model.cpu()
            model.reset_decoder_parameters()
            logging.info("Resetting model decoder...")
            if ngpus_per_node > 0:
                model = torch.nn.DataParallel(model).cuda()
                model_object = model.module
            else:
                model.cuda()
                model_object = model
            optimizer = optim.Adam(model.parameters(), lr=lr)
            variational = True

        if model_object.vae is not None and kl_anneal and variational:
            model_object.vae.beta = min(model_object.vae.beta + kl_rate,
                                        target_beta)

        if main_worker and ((i+1) % visualize_every == 0 or i == 0):
            logging.info(f"Epoch {i} (Train): %.4f", total_loss / total_item)
            model.eval()
            annotations = train.annotations
            reannotations = [{'desc': a['desc'].replace('blue', 'red').replace('yellow', 'green'),
                              'image': a['image']} for a in annotations]
            train.annotations = reannotations
            train_vis = visualize_filter_preds(model,
                                               train,
                                               vis_folder,
                                               prefix=f"train-{i}",
                                               gpu=gpu,
                                               n_eval=10)

            train.annotations = annotations
            test_vis = visualize_filter_preds(model,
                                              test,
                                              vis_folder,
                                              prefix=f"test-{i}",
                                              gpu=gpu)

            prior_vis = visualize_filter_preds(model,
                                               test,
                                               vis_folder,
                                               prefix=f"prior-{i}",
                                               gpu=gpu,
                                               prior=True)

            render_html(train_vis+test_vis+prior_vis, vis_folder)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=os.path.join(vis_folder, 'checkpoint.pth.tar'))


def visualize_filter_preds(model,
                           test,
                           vis_folder,
                           prefix="train-0",
                           prior=False,
                           gpu=0,
                           n_eval=5):

    test_loader = DataLoader(test,
                             batch_size=n_eval,
                             shuffle=True,
                             pin_memory=True,
                             collate_fn=test.collate)

    visualizations = []
    cmd, img, img_files = next(iter(test_loader))
    cmd = cmd.transpose(0, 1)
    if gpu is not None:
        cmd = cmd.cuda(gpu, non_blocking=True)
        img = img.cuda(gpu, non_blocking=True)
    _, _, _, *extras = model(**dict(cmd=cmd, img=img, test=True, prior=prior))
    results, attentions, text_attentions = map(utils.cpu, extras)
    cmd = utils.cpu(cmd).numpy()
    img = utils.cpu(img).numpy()
    for j in range(n_eval):
        visualizations.append(visualize(
            test,
            f"{prefix}-{j}",
            img_files[j],
            test.decode(cmd[j, :]),
            results,
            attentions,
            text_attentions,
            vis_folder
        ))
    return visualizations


def visualize(dataset,
              name,
              original,
              command,
              results,
              attentions,
              text_attentions,
              vis_folder):

    def prep(img, *, mean=[0.], std=[0.], transform=None):
        if img.shape[0] == 3:
            img = img * std[:, None, None] + mean[:, None, None]
            img = torch.clip(img, 0, 1)
        if transform is not None:
            img = transform(img)
        else:
            img = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            if img.shape[2] == 1:
                img = img[:, :, 0]
        return img

    # transform = torchvision.transforms.ToPILImage(mode=dataset.color)

    j = int(name.split('-')[-1])

    imprep = functools.partial(prep, mean=dataset.mean, std=dataset.std)

    imageio.imwrite(os.path.join(vis_folder, name + ".result-0.png"),
                    imprep(results[0][j, ...]))

    shutil.copyfile(original, os.path.join(vis_folder, name + ".original.png"))

    example = {
        "name": name,
        "command": command,
        "original": name + ".original.png",
        "init-result": name + ".result-0.png",
        "attentions": [],
        "text_attentions": [],
        "results": [],
    }

    for i in range(1, len(results)):
        example["attentions"].append(name + f".att-{i}.png")
        t_att = text_attentions[i-1][j, :].numpy().ravel().tolist()
        # t_att = " | ".join([f"{a:.1f}" for a in t_att])
        example["text_attentions"].append(t_att)
        example["results"].append(name + f".result-{i}.png")
        imageio.imwrite(os.path.join(vis_folder, name + f".att-{i}.png"),
                        prep(attentions[i-1][j, ...]))
        imageio.imwrite(os.path.join(vis_folder, name + f".result-{i}.png"),
                        imprep(results[i][j, ...]))
    return example


def render_html(visualizations, vis_folder):
    prefix = """<!DOCTYPE html>
                <html>
                <head>
                    <style>
                        .hovering{
                           height: 32px;
                        }
                		.hovering:hover {
                           transform: scale(4.0);
                        }
                </style>
                </head>
                <body>
            """
    suffix = """
                </body>
                </html>
             """
    with open(os.path.join(vis_folder, 'index.html'), "w") as writer:
        writer = functools.partial(print, file=writer)
        writer(prefix)
        for vis in visualizations:
            command = vis["command"]
            writer("<p>", vis["name"], command, "</p>")
            writer("<p>",
                   "<img class='hovering' src='",
                   vis['original'],
                   "'>", "</p>")
            writer("<table>")
            init = vis["init-result"]
            writer(f"<tr><td></td><td><img class='hovering' src='{init}'></td></tr>")
            for att, result, t_att in zip(vis["attentions"],
                                          vis["results"],
                                          vis["text_attentions"]):
                t_att = [f"{score:.1f} ({t}-{command[t]})" for (t, score) in enumerate(t_att) if score >= 0.1]
                t_att = " | ".join(t_att)
                writer(f"<tr><td><img class='hovering' src='{att}'></td><td><img class='hovering' src='{result}'></td><td>{t_att}</td></tr>")
            writer("</table>")
            writer("<br>")
        writer(suffix)


def filter_model(gpu, ngpus_per_node, args):
    args.gpu = gpu
    print(f"myrank: {args.rank} mygpu: {gpu}")
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + args.gpu
        utils.init_process(backend=args.dist_backend,
                           init_method=args.dist_url,
                           size=args.world_size,
                           rank=args.rank)

    if args.datatype == "setpp":
        train = SetDataset("data/setpp/", split="train")
        test = SetDataset("data/setpp/",
                          split="test",
                          transform=train.transform,
                          vocab=train.vocab)
    elif args.datatype == "shapes":
        train = ShapeDataset(root="data/shapes/", split="train")
        test = ShapeDataset(root="data/shapes/",
                            split="test",
                            transform=train.transform,
                            vocab=train.vocab)
    elif args.datatype == "clevr":
        train = CLEVRDataset("data/clevr/", split="trainA")
        test = CLEVRDataset("data/clevr/",
                            split="valB",
                            transform=train.transform,
                            vocab=train.vocab)
    else:
        train = SCANDataset("data/scan/", split="train")
        test = SCANDataset("data/scan/",
                           split="test",
                           transform=train.transform,
                           vocab=train.vocab)

    vis_folder = flags_to_path()
    os.makedirs(vis_folder, exist_ok=True)
    logging.info("vis folder: %s", vis_folder)

    if args.lex_vae_type == 'VAE':
        vae = VAE(3,
                  FLAGS.h_dim,
                  FLAGS.n_latent,
                  beta=FLAGS.beta,
                  size=train.size)
    elif args.lex_vae_type == 'VQVAE':
        vae = VectorQuantizedVAE(3,
                                 FLAGS.h_dim,
                                 FLAGS.n_latent,
                                 n_codes=FLAGS.n_codes,
                                 cc=FLAGS.commitment_cost,
                                 decay=FLAGS.decay,
                                 epsilon=FLAGS.epsilon,
                                 beta=FLAGS.beta,
                                 cmdproc=False,
                                 size=train.size,
                                 ).to(device)
    elif args.lex_vae_type == 'None':
        vae = None
    else:
        raise ValueError(f"Unknown vae type {FLAGS.lex_vae_type}")

    model = FilterModel(
        vocab=train.vocab,
        n_downsample=args.lex_n_downsample,
        n_latent=args.lex_n_latent,
        n_steps=args.lex_n_steps,
        att_loss_weight=args.lex_att_loss_weight,
        vae=vae,
        text_conditional=args.lex_text_conditional,
        append_z=args.lex_append_z,
    )

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            print('single gpu per process')
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.n_batch = int(args.n_batch / ngpus_per_node)
            args.n_workers = int((args.n_workers + ngpus_per_node - 1) / ngpus_per_node)
            model = DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            print('using DDP')
            model = DistributedDataParallel(model)
    elif args.gpu is not None:
        logging.info('using single gpu')
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print('using data parallel')
        model = torch.nn.DataParallel(model).cuda()


    train_filter_model(model,
                       train,
                       test,
                       vis_folder,
                       visualize_every=args.visualize_every,
                       n_batch=args.n_batch,
                       epoch=args.n_epoch,
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



def train_vae():
    if FLAGS.datatype == "setpp":
        train = SetDataset("data/setpp/", split="train")
        test = SetDataset("data/setpp/",
                          split="test",
                          transform=train.transform,
                          vocab=train.vocab)
    elif FLAGS.datatype == "shapes":
        train = ShapeDataset("data/shapes/", split="train")
        test = ShapeDataset("data/shapes/",
                            split="test",
                            transform=train.transform,
                            vocab=train.vocab)
    elif FLAGS.datatype == "clevr":
        train = CLEVRDataset("data/clevr/", split="trainA")
        test = CLEVRDataset("data/clevr/",
                            split="valB",
                            transform=train.transform,
                            vocab=train.vocab)
    else:
        train = SCANDataset("data/scan/", split="train")
        test = SCANDataset("data/scan/",
                           split="test",
                           transform=train.transform,
                           vocab=train.vocab)

    vis_folder = flags_to_path()
    os.makedirs(vis_folder, exist_ok=True)
    logging.info("vis folder: %s", vis_folder)

    if FLAGS.modeltype == "VQVAE":
        model = VectorQuantizedVAE(3, FLAGS.h_dim,
                                   FLAGS.n_latent,
                                   n_codes=FLAGS.n_codes,
                                   cc=FLAGS.commitment_cost,
                                   decay=FLAGS.decay,
                                   epsilon=FLAGS.epsilon,
                                   beta=FLAGS.beta,
                                   cmdproc=False,
                                   size=train.size,
                                   ).to(device)
    elif FLAGS.modeltype == "VAE":
        model = VAE(3,
                    FLAGS.h_dim,
                    FLAGS.n_latent,
                    beta=FLAGS.beta,
                    size=train.size).to(device)
    elif FLAGS.modeltype == "DAE":
        model = DAE(3,
                    latentdim=FLAGS.n_latent*4*4,
                    size=train.size).to(device)
    else:
        raise ValueError(f"Unknown model type {FLAGS.modeltype}")

    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)

    train_res_recon_error = 0.
    cnt = 0.

    loader = DataLoader(train,
                        batch_size=FLAGS.n_batch,
                        shuffle=True,
                        collate_fn=train.collate,
                        num_workers=FLAGS.n_workers,
                        worker_init_fn=utils.worker_init_fn)

    test_loader = DataLoader(test,
                             batch_size=32,
                             shuffle=True,
                             collate_fn=train.collate,
                             num_workers=FLAGS.n_workers,
                             worker_init_fn=utils.worker_init_fn)

    generator = iter(loader)

    model.train()

    for i in range(FLAGS.n_iter):
        try:
            cmd, img, _ = next(generator)
        except StopIteration:
            generator = iter(loader)
            cmd, img, _ = next(generator)

        img = img.to(device)
        cmd = cmd.to(device)
        optimizer.zero_grad()
        loss, _, recon_error, *_ = model(img, cmd)
        loss.backward()
        optimizer.step()

        train_res_recon_error += recon_error.item()
        cnt += img.shape[0]
        # train_res_nll.append(nll.item())

        if (i+1) % 1000 == 0:
            with torch.no_grad():
                logging.info('%d iterations', (i+1))
                logging.info('recon_error: %.6f', (train_res_recon_error / cnt))
                # #print('nll: %.6f' % np.mean(train_res_nll[-100:]))
                # print(len(test_loader))
                # val_recon, val_perp = evaluate_vqvae(model, test_loader)
                # print('val_recon_error: %.6f' % val_recon)
                # print('val_nll: %.6f' % val_perp)
                if (i+1) % 1000 == 0:
                    T = torchvision.transforms.ToPILImage(mode=train.color)
                    test_iter = iter(test_loader)
                    for j in range(5):
                        cmd, img, _ = next(test_iter)
                        img = img.to(device)
                        cmd = cmd.to(device)
                        loss, recon, recon_error, *_ = model(img, cmd)
                        recon = recon.cpu().data * train.std[None, :, None, None] + train.mean[None, :, None, None]
                        img = img.cpu().data * train.std[None, :, None, None] + train.mean[None, :, None, None]
                        res = torch.cat((recon, img), 0).clip_(0, 1)
                        T(make_grid(res)).convert("RGB").save(os.path.join(vis_folder, f"{i}_{j}.png"))
                        sample, _ = model.sample(B=32)
                        sample = sample.cpu().data * train.std[None, :, None, None] + train.mean[None, :, None, None]
                        T(make_grid(sample.clip_(0, 1))).convert("RGB").save(os.path.join(vis_folder, f"prior_{i}_{j}.png"))

    torch.save(model, os.path.join(vis_folder, "model.pt"))


def train_cvae():
    assert FLAGS.vae_path != ""

    if FLAGS.datatype == "setpp":
        train = SetDataset("data/setpp/", split="train")
        test = SetDataset("data/setpp/",
                          split="test",
                          transform=train.transform,
                          vocab=train.vocab)
    elif FLAGS.datatype == "shapes":
        train = ShapeDataset("data/shapes/", split="train")
        test = ShapeDataset("data/shapes/",
                            split="test",
                            transform=train.transform,
                            vocab=train.vocab)
    elif FLAGS.datatype == "clevr":
        train = CLEVRDataset("data/clevr/", split="trainA")
        test = CLEVRDataset("data/clevr/",
                            split="valB",
                            transform=train.transform,
                            vocab=train.vocab)
    else:
        train = SCANDataset("data/scan/", split="train")
        test = SCANDataset("data/scan/",
                           split="test",
                           transform=train.transform,
                           vocab=train.vocab)

    vis_folder = flags_to_path()
    os.makedirs(vis_folder, exist_ok=True)
    logging.info("vis folder: %s", vis_folder)

    if FLAGS.test:
        model = torch.load(FLAGS.model_path)
        model = model.to(device)

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
                           ).to(device)
            model.vqvae.load_state_dict(torch.load(FLAGS.vae_path).state_dict())
            for p in model.vqvae.parameters():
                p.requires_grad = False
            if FLAGS.lex_path != "":
                model.set_src_copy(model.get_src_lexicon(FLAGS.lex_path))

            model = model.to(device)

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
                cmd = cmd.to(device)
                img = img.to(device)
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
            model = model.to(device)
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

            cmd = cmd.to(device)
            img = img.to(device)

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
                        cmd = cmd.to(device)
                        img = img.to(device)
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
            cmd = cmd.to(device)
            img = img.to(device)
            logging.info("sampling")
            recon, *_ = model.predict(cmd, top_k=10, sample=True)
            logging.info("sampled")
            recon = recon.cpu().data * train.std[None, :, None, None] + train.mean[None, :, None, None]
            img = img.cpu().data * train.std[None, :, None, None] + train.mean[None, :, None, None]
            res = torch.cat((recon, img), 0).clip_(0, 1)
            logging.info("saving samples")
            T(make_grid(res, nrow=res.shape[0]//2)).convert("RGB").save(os.path.join(vis_folder, f"eval_{j}.png"))
            logging.info("saved")


def evaluate_vqvae(model, loader):
    val_res_recon_error = 0.0
    val_res_nll = 0.0
    cnt = 0
    for (cmd, img, _) in loader:
        img = img.to(device)
        cmd = cmd.to(device)
        cnt += img.shape[0]
        _, _, recon_error, *_ = model(img, cmd)
        nll = model.nll(img, cmd)
        val_res_recon_error += recon_error.item()
        val_res_nll += nll.item()
    model.train()
    return val_res_recon_error / cnt, val_res_nll / cnt


def evaluate_cvae(model,loader):
    val_recon_error = 0.0
    val_loss = 0.0
    cnt = 0
    for (cmd, img, _) in loader:
        img = img.to(device)
        cmd = cmd.to(device)
        cnt += img.shape[0]
        loss = model(img, cmd)
        x_tilde, *_ = model.predict(cmd)
        val_recon_error += (x_tilde-img).pow(2).sum().item()
        val_loss += loss.item() * img.shape[0]
        if cnt > 100:
            break
    return val_recon_error / cnt, val_loss / cnt


def img2code():

    if FLAGS.datatype == "setpp":
        train = SetDataset("data/setpp/", split="train")
        test = SetDataset("data/setpp/",
                          split="test",
                          transform=train.transform,
                          vocab=train.vocab)
    elif FLAGS.datatype == "shapes":
        train = ShapeDataset("data/shapes/", split="train")
        test = ShapeDataset("data/shapes/",
                            split="test",
                            transform=train.transform,
                            vocab=train.vocab)
    elif FLAGS.datatype == "clevr":
        train = CLEVRDataset("data/clevr/", split="trainA")
        test = CLEVRDataset("data/clevr/",
                            split="valB",
                            transform=train.transform,
                            vocab=train.vocab)
    else:
        train = SCANDataset("data/scan/", split="train")
        test = SCANDataset("data/scan/",
                           split="test",
                           transform=train.transform,
                           vocab=train.vocab)

    vis_folder = flags_to_path()
    os.makedirs(vis_folder, exist_ok=True)
    logging.info("vis folder: %s", vis_folder)

    if FLAGS.modeltype == "VQVAE":
        model = VectorQuantizedVAE(3,
                                   FLAGS.h_dim,
                                   FLAGS.n_latent,
                                   n_codes=FLAGS.n_codes,
                                   cc=FLAGS.commitment_cost,
                                   decay=FLAGS.decay,
                                   epsilon=FLAGS.epsilon,
                                   beta=FLAGS.beta,
                                   cmdproc=False,
                                   size=train.size).to(device)
        model.load_state_dict(torch.load(FLAGS.vae_path).state_dict())
        model.eval()
    else:
        raise ValueError(f"model type not available for this {FLAGS.modeltype}")

    train_loader = DataLoader(train,
                              batch_size=FLAGS.n_batch,
                              shuffle=False,
                              collate_fn=train.collate,
                              num_workers=FLAGS.n_workers)
    test_loader = DataLoader(test,
                             batch_size=FLAGS.n_batch,
                             shuffle=False,
                             collate_fn=train.collate,
                             num_workers=FLAGS.n_workers)
    for (split, loader) in zip(("train", "test"), (train_loader, test_loader)):
        generator = iter(loader)
        with open(os.path.join(vis_folder, f"{split}_encodings.txt"), "w") as f:
            for i in range(len(generator)):
                try:
                    cmd, img, names = next(generator)
                except StopIteration:
                    generator = iter(loader)
                    cmd, img, names = next(generator)

                img = img.to(device)
                cmd = cmd.to(device)
                _, _, recon_error, _, _, encodings = model(img, cmd)
                for k in range(img.shape[0]):
                    encc = encodings[k].flatten().detach().cpu().numpy()
                    encc = [str(e) for e in encc]
                    cmdc = train.vocab.decode_plus(cmd[:, k].detach().cpu().numpy())
                    line = " ".join(cmdc) + "\t" + " ".join(encc) + "\t" + names[k] + "\n"
                    f.write(line)


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

def main(_):
    if FLAGS.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        logging.tb_writer = SummaryWriter(os.path.join(FLAGS.tensorboard,
                                                       FLAGS.datatype,
                                                       FLAGS.modeltype,
                                                       (f"dim_{FLAGS.n_latent}_"
                                                        f"lr_{FLAGS.lr}_"
                                                        f"beta_{FLAGS.beta}")
                                                       ))

    if FLAGS.rank == 0:
        logging._absl_handler.setFormatter(logging.logging.Formatter(
                    '%(asctime)s [%(filename)s:%(lineno)d] %(message)s', "%H:%M:%S"))

    if FLAGS.seed is not None:
        utils.set_seed(FLAGS.seed + FLAGS.rank)

    if FLAGS.filter_model:
        if FLAGS.gpu is not None:
            logging.warning('You have chosen a specific GPU. This '
                         'will completely disable data parallelism.')

        if FLAGS.dist_url == "env://" and FLAGS.world_size == -1:
            FLAGS.world_size = int(os.environ["WORLD_SIZE"])

        FLAGS.distributed = FLAGS.world_size > 1 or FLAGS.multiprocessing_distributed

        ngpus_per_node = torch.cuda.device_count()

        args = utils.ConfigDict(dict((item.name, item.value)
                      for item in FLAGS.get_key_flags_for_module(sys.argv[0])))

        logging.tb_writer.add_text("FLAGS", json.dumps(args._initial_dict, indent=2).replace("\n", "   \n"))

        if args.multiprocessing_distributed:
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            args.world_size = ngpus_per_node * args.world_size
            # Use torch.multiprocessing.spawn to launch distributed processes: the
            # main_worker process function
            torch.multiprocessing.spawn(filter_model,
                                        nprocs=ngpus_per_node,
                                        args=(ngpus_per_node, args))
        else:
            # Simply call main_worker function
            filter_model(FLAGS.gpu, ngpus_per_node, args)
    elif FLAGS.extract_codes:
        img2code()
    elif FLAGS.modeltype.startswith("C"):
        train_cvae()
    else:
        train_vae()

    if FLAGS.tensorboard:
        logging.tb_writer.close()


if __name__ == "__main__":
    logging.info(f"DEVICE: {device}")
    app.run(main)
