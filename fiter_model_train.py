import os
import functools
import imageio
import shutil

import numpy as np
import torch

from absl import app, flags, logging
from tqdm import tqdm

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import options
from src import utils
from src.lex import FilterModel
from src.vqvae import VectorQuantizedVAE
from src.vae import VAE
from src import parallel
from src.datasets import get_data


FLAGS = flags.FLAGS


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


def train_filter_model(model,
                       train,
                       val,
                       test,
                       vis_folder,
                       optimizer=None,
                       start_epoch=0,
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

    if optimizer is None:
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
        model_object = model

    if model_object.vae is not None:
        variational = True
        if kl_anneal:
            target_beta = model_object.vae.beta
            model_object.vae.beta = 0.0
            kl_rate = 4*(target_beta / (epoch))
            logging.info(f"kl rate: {kl_rate}")
        if decoder_reset != -1:
            variational = False

    writer = utils.get_tensorboard_writer()

    for i in range(start_epoch, epoch):

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

            loss, scalars = model(
                **dict(cmd=cmd, img=img, test=False, variational=variational))
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
                    writer.add_scalars('Train/Loss', scalars,
                                                  total_steps)
                    writer.add_scalar('beta', model_object.vae.beta,
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
            reannotations = [{'text': a['text'].replace('blue', 'green'),
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

            utils.save_checkpoint({
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
    args.ngpus_per_node = ngpus_per_node
    parallel.init_distributed(args)

    train, val, test = get_data()
    vis_folder = utils.flags_to_path()
    os.makedirs(vis_folder, exist_ok=True)
    logging.info("vis folder: %s", vis_folder)

    if args.lex_vae_type == 'VAE':
        vae = VAE(3,
                  args.h_dim,
                  args.n_latent,
                  beta=args.beta,
                  size=train.size)
    elif args.lex_vae_type == 'VQVAE':
        vae = VectorQuantizedVAE(3,
                                 args.h_dim,
                                 args.n_latent,
                                 n_codes=args.n_codes,
                                 cc=args.commitment_cost,
                                 decay=args.decay,
                                 epsilon=args.epsilon,
                                 beta=args.beta,
                                 cmdproc=False,
                                 size=train.size,
                                 )
    elif args.lex_vae_type == 'None':
        vae = None
    else:
        raise ValueError(f"Unknown vae type {args.lex_vae_type}")

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

    model = parallel.distribute(model, args)

    args.start_epoch = 0

    optimizer, scheduler = utils.resume(model, args)

    train_filter_model(model,
                       train,
                       val,
                       test,
                       vis_folder,
                       optimizer=optimizer,
                       start_epoch=args.start_epoch,
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


def main(_):
    parallel.run_parallel(filter_model)


if __name__ == "__main__":
    app.run(main)
