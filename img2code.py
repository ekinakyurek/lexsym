import os
import torch

from absl import app, flags, logging

from torch import optim
from torch.utils import data as torch_data
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import options
import vae_train
from src import utils
from src.lex import FilterModel
from src.vqvae import VectorQuantizedVAE
from src.vae import VAE
from src.dae import DAE
from src import parallel
from src.datasets import get_data


FLAGS = flags.FLAGS


def img2code(model,
             train,
             test,
             vis_folder,
             n_batch=64,
             n_workers=4,
             distributed=False,
             ngpus_per_node=1,
             gpu=0,
             rank=0):

    main_worker = rank % ngpus_per_node == 0

    if not main_worker:
        return -1

    logging.info(ngpus_per_node)

    train_loader = DataLoader(train,
                              batch_size=n_batch,
                              shuffle=False,
                              collate_fn=train.collate,
                              num_workers=n_workers)

    test_loader = DataLoader(test,
                             batch_size=n_batch,
                             shuffle=False,
                             collate_fn=train.collate,
                             num_workers=n_workers)

    for (split, loader) in zip(("train", "test"), (train_loader, test_loader)):
        generator = iter(loader)
        with open(os.path.join(vis_folder, f"{split}_encodings.txt"), "w") as f:
            for _ in range(len(generator)):
                try:
                    cmd, img, names = next(generator)
                except StopIteration:
                    generator = iter(loader)
                    cmd, img, names = next(generator)

                cmd = cmd.transpose(0, 1)
                if gpu is not None:
                    cmd = cmd.cuda(gpu, non_blocking=True)
                    img = img.cuda(gpu, non_blocking=True)

                _, _, recon_error, _, _, encodings = model(**dict(img=img, cmd=cmd))

                for k in range(img.shape[0]):
                    encc = encodings[k].flatten().detach().cpu().numpy()
                    encc = [str(e) for e in encc]
                    cmdc = train.vocab.decode_plus(cmd[k, :].detach().cpu().numpy())
                    line = " ".join(cmdc) + "\t" + " ".join(encc) + "\t" + names[k] + "\n"
                    f.write(line)


def img2code_runner(gpu, ngpus_per_node, args):
    args.gpu = gpu
    args.ngpus_per_node = ngpus_per_node
    parallel.init_distributed(args)

    train, test = get_data()
    vis_folder = utils.flags_to_path()
    os.makedirs(vis_folder, exist_ok=True)
    logging.info("vis folder: %s", vis_folder)

    if args.modeltype == "VQVAE":
        model = VectorQuantizedVAE(3,
                                   args.h_dim,
                                   args.n_latent,
                                   n_codes=args.n_codes,
                                   cc=args.commitment_cost,
                                   decay=args.decay,
                                   epsilon=args.epsilon,
                                   beta=args.beta,
                                   cmdproc=False,
                                   size=train.size)
    else:
        raise ValueError(f"model type not available for this {args.modeltype}")

    model = parallel.distribute(model, args)

    if not hasattr(model, 'sample'):
        model.sample = model.module.sample

    args.start_iter = 0

    utils.resume(model, args, mark='iter')

    img2code(model,
             train,
             test,
             vis_folder,
             n_batch=args.n_batch,
             n_workers=args.n_workers,
             distributed=args.distributed,
             ngpus_per_node=ngpus_per_node,
             gpu=gpu,
             rank=args.rank)


def main(_):
    parallel.run_parallel(img2code_runner)


if __name__ == "__main__":
    app.run(main)
