import os
import sys
import json
import torch
from torch import optim
from src import utils
from absl import flags
from absl import logging
from torch.nn.parallel import DistributedDataParallel

FLAGS = flags.FLAGS

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



def run_parallel(runner_fn):
    writer = utils.get_tensorboard_writer()

    if FLAGS.rank == 0:
        utils.set_logging_format()

    if FLAGS.seed is not None:
        utils.set_seed(FLAGS.seed + FLAGS.rank)

    if FLAGS.gpu is not None:
        logging.warning('You have chosen a specific GPU. This '
                        'will completely disable data parallelism.')

    if FLAGS.dist_url == "env://" and FLAGS.world_size == -1:
        FLAGS.world_size = int(os.environ["WORLD_SIZE"])

    FLAGS.distributed = FLAGS.world_size > 1 or FLAGS.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    args = utils.flags_to_args()

    print(json.dumps(args._initial_dict, indent=2))

    # st_args = json.dumps(args._initial_dict, indent=2).replace("\n", "   \n")
    #
    # writer.add_text("FLAGS", st_args)

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        torch.multiprocessing.spawn(runner_fn,
                                    nprocs=ngpus_per_node,
                                    args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        runner_fn(FLAGS.gpu, ngpus_per_node, args)

    writer.close()


def distribute(model, args):
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            print('single gpu per process')
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.n_batch = int(args.n_batch / args.ngpus_per_node)
            args.n_workers = int((args.n_workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
            model = DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            print('using DDP')
            model = DistributedDataParallel(model)
    elif args.gpu is not None:
        print('using single gpu')
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print('using data parallel')
        model = torch.nn.DataParallel(model).cuda()
    return model


def init_distributed(args):
    print(f"myrank: {args.rank} mygpu: {args.gpu}")
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * args.ngpus_per_node + args.gpu
        utils.init_process(backend=args.dist_backend,
                           init_method=args.dist_url,
                           size=args.world_size,
                           rank=args.rank)
