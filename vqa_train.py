import os
import json
import functools
import torch
from seq2seq import hlog
from absl import app, flags
from tqdm import tqdm

import numpy as np
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image

import options
from src import utils
from src import lexutils
from src import parallel
from src.vqa import VQA
from src.datasets import get_data
from seq2seq.src import NoamLR

FLAGS = flags.FLAGS

flags.DEFINE_string("modeltype", default='VQA',
                    help='VQA')

flags.DEFINE_string('vae_path', default=None,
                    help='A pretrained vae path for conditional vae models.')

flags.DEFINE_string("lex_and_swaps_path", default=None,
                    help='A prelearned lexicon path to be used in text-image '
                         'vqvae models')

flags.DEFINE_string("code_files", default=None,
                    help='Pre cached codes for images')

flags.DEFINE_string("imgsize", default='128,128',
                    help='resize dimension for input images')

def unnormalize(img, std, mean):
    return img * std[None, :, None, None] + mean[None, :, None, None]


def decode_datum(question, answer, pred, vocab, answer_vocab):
    question = " ".join(vocab.decode_plus(question))
    answer = answer_vocab._rev_contents[answer]
    pred = answer_vocab._rev_contents[pred]
    return question, answer, pred

## add categories to the dataset
## print to a file
## remove printing images
def evaluate_model(model,
                   test_loader,
                   vis_folder,
                   code_cache=None,
                   niter=0,
                   gpu=None,
                   writer=None,
                   split="val"):

    vocab, answer_vocab = test_loader.dataset.vocab, test_loader.dataset.answer_vocab

    total = .0
    nlls = []
    accs = []
    predictions = []

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

        accs.append((pred == answer).sum().item())
        nlls.append(nll.mean().item() * len(answer))
        total += len(answer)
    
        if writer is not None:
            for i in range(img.shape[0]):
                datum = decode_datum(question[i].cpu().numpy(),
                                     answer[i].item(),
                                     pred[i].item(),
                                     vocab,
                                     answer_vocab)

                question_i, answer_i, pred_i = datum
                file_i = files[i]
                line = f"{question_i}\t{answer_i}\t{pred_i}\t{file_i}"
                predictions.append(line)
   
    mean_nll = np.sum(nlls) / total
    mean_acc = np.sum(accs) / total

    if writer is not None:
        with open(os.path.join(vis_folder, f'predictions_{split}_{niter}.txt'), "w") as f:
            f.write("\n".join(predictions))
            
        writer.add_scalar(f"Accuracy/{split}", mean_acc, niter)
        writer.add_scalar(f"Loss/Test/{split}", mean_nll, niter)

    return mean_nll, mean_acc


def train_vqa_model_model(model,
                          train,
                          val,
                          test,
                          vis_folder,
                          scheduler=None,
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
                          lr=0.0001,
                          gclip=-1,
                          gaccum=1):

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    if warmup_steps != -1 and scheduler is None:
        try:
            rnn_dim = model.rnn_dim
        except:
            rnn_dim = model.module.rnn_dim
            
        scheduler = NoamLR(optimizer, rnn_dim, warmup_steps=warmup_steps)

    main_worker = rank == 0 if distributed else True
    print(f"rank: {rank} starts training")
    
    hlog.log(ngpus_per_node)

    train_sampler = DistributedSampler(train, shuffle=True, seed=0) if distributed else None
 
    worker_init_fn = functools.partial(utils.worker_init_fn, rank=rank)

    train_loader = DataLoader(train,
                              batch_size=n_batch,
                              shuffle=(train_sampler is None),
                              collate_fn=train.collate,
                              sampler=train_sampler,
                              num_workers=n_workers,
                              drop_last=gaccum > 1 or (train_sampler is not None),
                              worker_init_fn=worker_init_fn)

    test_loader = DataLoader(test,
                             batch_size=n_batch,
                             shuffle=True,
                             collate_fn=train.collate,
                             num_workers=n_workers,
                             worker_init_fn=utils.worker_init_fn)
    
    val_loader = DataLoader(val,
                            batch_size=n_batch,
                            shuffle=True,
                            collate_fn=train.collate,
                            num_workers=n_workers,
                            worker_init_fn=utils.worker_init_fn)

    if main_worker:
        writer = utils.get_tensorboard_writer(vis_folder)

    total, nll = 0., 0.
    epoch = 0
    
    if train_sampler:
        train_sampler.set_epoch(epoch)  # FIX
        
    train_iter = iter(train_loader)
    model.train()
    for i in range(start_iter, n_iter * gaccum):
        # Get data
        try:
            question, img, answer, files = next(train_iter)
        except StopIteration:
            epoch += 1
            if train_sampler:
                train_sampler.set_epoch(epoch)  # FIX
            train_iter = iter(train_loader)
            question, img, answer, files = next(train_iter)

        question = question.transpose(0, 1)

        # Get img codes if precached
        if code_cache is not None:
            img = torch.stack([code_cache[f] for f in files], dim=0)
            if lexicon is not None:
                if i == 0:
                    hlog.log("Doing random  swaps!")
                lexutils.random_swap(lexicon,
                                     question,
                                     train.vocab,
                                     answer,
                                     train.answer_vocab,
                                     img)
        # Send data to gpu
        if gpu is not None:
            img = img.cuda(gpu, non_blocking=True)
            question = question.cuda(gpu, non_blocking=True)
            answer = answer.cuda(gpu, non_blocking=True)

        if i % gaccum == 0:
            optimizer.zero_grad(set_to_none=True)
            
        # Take grad step on loss
        loss = model(**dict(question=question, img=img, answer=answer))
        loss = loss.mean() / gaccum
        loss.backward()
        
        if (i+1) % gaccum == 0:
            if gclip != -1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gclip)
            optimizer.step()

            # Step the scheduler
            if scheduler is not None:
                scheduler.step()

        with torch.no_grad():
            # Accumulate metrics
            total += img.shape[0]
            nll += loss.item()

        # Evaluate and display metrics
        if main_worker and (i+1) % (visualize_every * gaccum) == 0:
            with torch.no_grad():
                hlog.log('%d gradient updates' % ((i+1) / gaccum))
                hlog.log('Train Loss: %.6f' % (nll / total))
                if writer is not None:
                    writer.add_scalar('Loss/Train', nll/total, i+1)
                model.eval()
                val_nll, val_acc = evaluate_model(model,
                                                  val_loader,
                                                  vis_folder,
                                                  code_cache,
                                                  gpu=gpu,
                                                  writer=writer,
                                                  niter=i+1,
                                                  split="val")
                print(f"val_nll: {val_nll} val_acc: {val_acc}")
                test_nll, test_acc = evaluate_model(model,
                                                    test_loader,
                                                    vis_folder,
                                                    code_cache,
                                                    gpu=gpu,
                                                    writer=writer,
                                                    niter=i+1,
                                                    split="test")
                print(f"test_nll: {test_nll} test_acc: {test_acc}")
                # Save checkpoint
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'iter': i,
                    'gaccum': gaccum,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, filename=os.path.join(vis_folder,
                                         'checkpoint.pth.tar'))
                model.train()
    

def train_vqa_model(gpu, ngpus_per_node, args):
    assert args.vae_path is not None
    vis_folder = utils.flags_to_path(args)
    os.makedirs(vis_folder, exist_ok=True)
    hlog.log("vis folder: %s" % vis_folder)

    args.gpu = gpu
    args.ngpus_per_node = ngpus_per_node
    parallel.init_distributed(args)
    
    img_size = tuple(map(int, args.imgsize.split(',')))
    
    code_cache = lexicon = None

    if args.code_files is not None:
        code_cache = {}
        files = args.code_files.split(',')
        for file in files:
            with open(file) as handle:
                for line in handle:
                    _, code, filename = line.split('\t')
                    code_cache[filename.strip()] = torch.tensor(
                                        list(map(int, code.split())))
        assert code_cache is not None
        hlog.log('using code cache')
        
    if args.lex_and_swaps_path is not None:
        with open(args.lex_and_swaps_path) as f:
            lexicon = json.load(f)
        hlog.log(f'Lex and Swaps:\n{lexicon}')

    train, val, test = get_data(datatype=args.datatype,
                                dataroot=args.dataroot,
                                vqa=True, no_images=code_cache is not None,
                                size=img_size)

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

    optimizer = None
    scheduler = None
    args.start_iter = 0
    
    if code_cache is None:
        args.resume = args.vae_path
        vqvae = nn.DataParallel(model.vqvae)
        utils.resume(vqvae, args, mark='iter', load_optims=False)
        model.vqvae = vqvae.module
        model = parallel.distribute(model, args)
        args.start_iter = 0
    else:
        model.vqvae = None
        
    if args.resume is not None:
        args.start_iter = 0
        model = parallel.distribute(model, args)
        optimizer, scheduler = utils.resume(model, args, mark='iter')
    else:
        model = parallel.distribute(model, args)

    train_vqa_model_model(model,
                          train,
                          val,
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
                          lr=args.lr,
                          gclip=args.gclip,
                          gaccum=args.gaccum)
    
    if args.distributed:
        parallel.cleanup()


def main(_):
    parallel.run_parallel(train_vqa_model)


if __name__ == "__main__":
    app.run(main)
