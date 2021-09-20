import os
import torch
import imageio
import json
import random
import numpy as np
from torch import optim
from torch.utils import data as torch_data
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from src.lex import FilterModel
from src.vqvae import VectorQuantizedVAE, CVQVAE
from src.vae import VAE, CVAE
from src.dae import DAE
from src import utils
# from src.utils import make_number_grid

from data.shapes import ShapeDataset
from data.set import SetDataset
from data.scan import SCANDataset
from data.clevr import CLEVRDataset
import torchvision
from torchvision.utils import make_grid, save_image
import itertools
import functools
import warnings
from absl import app, flags, logging
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FLAGS = flags.FLAGS

flags.DEFINE_integer("h_dim", 32, "")
flags.DEFINE_integer("rnn_dim", 256, "")
flags.DEFINE_integer("rnn_n_layers", 2, "")
flags.DEFINE_float("rnn_drop", 0.1, "")
flags.DEFINE_integer("n_latent", 24, "")
flags.DEFINE_integer("n_batch", 128, "")
flags.DEFINE_integer("n_iter", 100000, "")
flags.DEFINE_integer("n_epoch", 50, "")
flags.DEFINE_integer("n_codes", 10, "")
flags.DEFINE_integer("n_workers", 4, "")
flags.DEFINE_integer("seed", 0, "")
flags.DEFINE_float("beta", 1.0, "")
flags.DEFINE_float("commitment_cost", 0.25, "")
flags.DEFINE_string("datatype", "setpp", "")
flags.DEFINE_string("modeltype", "VQVAE", "")
flags.DEFINE_float("decay", 0.99, "")
flags.DEFINE_float("lr", 1e-3, "")
flags.DEFINE_float("epsilon", 1e-5, "")
flags.DEFINE_bool("debug", False, "")
flags.DEFINE_bool("highdrop", False, "")
flags.DEFINE_bool("highdroptest", False, "")
flags.DEFINE_float("highdropvalue", 0., "")
flags.DEFINE_bool("copy", False, "")
flags.DEFINE_string("vae_path", "", "pretrained vae path")
flags.DEFINE_string("lex_path", "", "prelearned lexicon")
flags.DEFINE_string("model_path", "", "prelearned model")
flags.DEFINE_bool("extract_codes", False, "")
flags.DEFINE_bool("filter_model", False, "")
flags.DEFINE_bool("test", False, "")


flags.DEFINE_bool("distributed", False, "")

flags.DEFINE_integer('gpu', default=None,
                     help='GPU id to use.')

flags.DEFINE_integer('rank', default=0,
                     help='node rank for distributed training')

flags.DEFINE_integer('world_size', default=1,
                     help='ngpus')

flags.DEFINE_string('dist_backend', default='nccl',
                    help='distributed backend')

flags.DEFINE_string('dist_url', default='tcp://127.0.0.1:23456',
                    help='url used to set up distributed training')

flags.DEFINE_bool('multiprocessing_distributed', False,
                  help='Use multi-processing distributed training to launch '
                       'N processes per node, which has N GPUs. This is the '
                       'fastest way to use PyTorch for either single node or'
                       ' multi node data parallel training')


def train_filter_model(model,
                       train,
                       test,
                       optimizer,
                       vis_folder,
                       n_batch=64,
                       epoch=1,
                       n_workers=1,
                       distributed=False,
                       ngpus_per_node=1,
                       gpu=0,
                       rank=0):

    main_worker = rank % ngpus_per_node == 0

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

    for i in range(epoch):
        if train_sampler:
            train_sampler.set_epoch(i)
        total_loss = .0
        total_item = .0
        tloader = tqdm(loader) if main_worker else loader
        model.train()
        for (cmd, img, _) in tloader:
            cmd = cmd.transpose(0, 1)
            if gpu is not None:
                cmd = cmd.cuda(gpu, non_blocking=True)
                img = img.cuda(gpu, non_blocking=True)
            loss = model(cmd, img, test=False)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            total_loss += loss.mean().item()
            total_item += img.shape[0]
            if main_worker:
                tloader.set_description(
                 f"Epoch {i}, Avg Loss (Train): %.4f, N: %d" %
                 (total_loss/total_item, total_item))
        if main_worker:
            logging.info(f"Epoch {i} (Train): %.4f", total_loss / total_item)
            model.eval()
            train_vis = visualize_filter_preds(model,
                                               train,
                                               vis_folder,
                                               gpu=gpu,
                                               n_workers=n_workers,
                                               it=i)

            test_vis = visualize_filter_preds(model,
                                              test,
                                              vis_folder,
                                              gpu=gpu,
                                              n_workers=n_workers,
                                              it=i)

            render_html(train_vis+test_vis, vis_folder)


def visualize_filter_preds(model,
                           test,
                           vis_folder,
                           gpu=0,
                           n_workers=1,
                           it=0,
                           n=3):

    test_loader = DataLoader(test,
                             batch_size=5,
                             shuffle=False,
                             pin_memory=True,
                             collate_fn=test.collate,
                             num_workers=n_workers,
                             worker_init_fn=utils.worker_init_fn)

    visualizations = []
    cmd, img, _ = next(iter(test_loader))
    cmd = cmd.transpose(0, 1)
    if gpu is not None:
        cmd = cmd.cuda(gpu, non_blocking=True)
        img = img.cuda(gpu, non_blocking=True)
    _, *extras = model(**dict(cmd=cmd, img=img, test=True))
    _, results, attentions, text_attentions = map(utils.cpu, extras)
    cmd = utils.cpu(cmd).numpy()
    img = utils.cpu(img).numpy()
    for j in range(n):
        visualizations.append(visualize(
            test,
            f"train-{it}-{j}",
            test.decode(cmd[j, :]),
            results,
            attentions,
            text_attentions,
            vis_folder
        ))
    return visualizations


def visualize(dataset,
              name,
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

    imageio.imwrite(os.path.join(vis_folder, name + ".result-{0}.png"),
                    imprep(results[0][j, ...]))

    example = {
        "name": name,
        "command": command,
        "init-result": name + ".result-0.png",
        "attentions": [],
        "text_attentions": [],
        "results": [],
    }
    for i in range(1, len(results)):
        example["attentions"].append(name + f".att-{i}.png")
        t_att = text_attentions[i-1][j, :].numpy().ravel().tolist()
        t_att = " | ".join([f"{a:.1f}" for a in t_att])
        example["text_attentions"].append(t_att)
        example["results"].append(name + f".result-{i}.png")
        imageio.imwrite(os.path.join(vis_folder, name + f".att-{i}.png"),
                        prep(attentions[i-1][j, ...]))
        imageio.imwrite(os.path.join(vis_folder, name + f".result-{i}.png"),
                        imprep(results[i][j, ...]))
    return example


def render_html(visualizations, vis_folder):
    with open(os.path.join(vis_folder, 'index.html'), "w") as writer:
        writer = functools.partial(print, file=writer)
        for vis in visualizations:
            writer("<p>", vis["name"], vis["command"], "</p>")
            writer("<table>")
            init = vis["init-result"]
            writer(f"<tr><td></td><td><img src='{init}' height=30></td></tr>")
            for att, result, t_att in zip(vis["attentions"], vis["results"], vis["text_attentions"]):
                writer(f"<tr><td><img src='{att}' height=30></td><td><img src='{result}' height=30></td><td>{t_att}</td></tr>")
            writer("</table>")
            writer("<br>")


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

    vis_folder = os.path.join("vis", args.datatype, "FilterModel")
    os.makedirs(vis_folder, exist_ok=True)
    logging.info("vis folder: %s", vis_folder)

    model = FilterModel(
        vocab=train.vocab,
        n_downsample=2,
        n_latent=args.n_latent,
        n_steps=10
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

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_filter_model(model,
                       train,
                       test,
                       optimizer,
                       vis_folder,
                       n_batch=args.n_batch,
                       epoch=args.n_epoch,
                       n_workers=args.n_workers,
                       distributed=args.distributed,
                       ngpus_per_node=ngpus_per_node,
                       gpu=args.gpu,
                       rank=args.rank)
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
                        recon = recon.cpu().data * train.std[None, :, None, None] + train.mean[None, abs:, None, None]
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
    root = os.path.join("vis", FLAGS.datatype, FLAGS.modeltype)

    if "VQVAE" in FLAGS.modeltype:
        return os.path.join(root,
                            (f"beta_{FLAGS.beta}_ncodes_{FLAGS.n_codes}_"
                             f"ldim_{FLAGS.n_latent}_dim_{FLAGS.h_dim}_"
                             f"lr_{FLAGS.lr}")
                            )
    else:
        return os.path.join(root,
                            (f"beta_{FLAGS.beta}_ldim_{FLAGS.n_latent}_"
                             f"dim_{FLAGS.h_dim}"
                             f"_lr_{FLAGS.lr}")
                            )


def main(_):
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

        if FLAGS.multiprocessing_distributed:
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            FLAGS.world_size = ngpus_per_node * FLAGS.world_size
            # Use torch.multiprocessing.spawn to launch distributed processes: the
            # main_worker process function
            args = utils.ConfigDict(
                              {k: v.value for (k, v) in FLAGS.__flags.items()})

            torch.multiprocessing.spawn(filter_model, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        else:
            # Simply call main_worker function
            filter_model(FLAGS.gpu, ngpus_per_node, FLAGS)
        return -1
    if FLAGS.extract_codes:
        return img2code()
    elif FLAGS.modeltype.startswith("C"):
        return train_cvae()
    else:
        return train_vae()


if __name__ == "__main__":
    logging.info(f"DEVICE: {device}")
    app.run(main)
