import os
import json
import torch
from src import utils
from absl import flags
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
from seq2seq import hlog


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


def get_dist_env():
    if 'OMPI_COMM_WORLD_SIZE' in os.environ:
        world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE'))
    else:
        world_size = int(os.getenv('SLURM_NTASKS'))

    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        global_rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
    else:
        global_rank = int(os.getenv('SLURM_PROCID'))
    init_meth = ''
    if 'MASTER_ADDR'in os.environ:
        init_meth ='tcp://'+ os.getenv('MASTER_ADDR')
    if 'MASTER_PORT'in os.environ:
        init_meth = init_meth + ':' + os.getenv('MASTER_PORT')
    print("init method: ", init_meth)
    print("word_size: ", world_size)
    print("global rank: ", global_rank)
    return global_rank, world_size, init_meth


def run_parallel(runner_fn):
    writer = utils.get_tensorboard_writer(utils.flags_to_path(FLAGS))

    if FLAGS.seed is not None:
        utils.set_seed(FLAGS.seed + FLAGS.rank)
        hlog.log(f"Setting seed to {FLAGS.seed}")

    if FLAGS.gpu is not None:
        hlog.log('You have chosen a specific GPU. This '
                 'will completely disable data parallelism.')

    # if FLAGS.dist_url == "env://" and FLAGS.world_size == -1:
    #     FLAGS.world_size = int(os.environ["WORLD_SIZE"])
    
    global_rank, world_size, init_meth = get_dist_env()
    
    FLAGS.world_size = world_size
    FLAGS.rank = global_rank
    FLAGS.dist_url = init_meth
    
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
        if args.distributed and args.gpu is None:
            args.gpu = args.rank % ngpus_per_node
        # Simply call main_worker function
        runner_fn(args.gpu, ngpus_per_node, args)

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
            args.n_batch = int(args.n_batch / args.world_size)
            print("n_batch_per_process: ", args.n_batch)
            args.n_workers = int((args.n_workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
            model = DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True, output_device=None)
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


def init_process(rank=0, size=1, backend='nccl', init_method="tcp://127.0.0.1:23456"):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend,
                            rank=rank,
                            world_size=size,
                            init_method=init_method)


def cleanup():
    dist.destroy_process_group()


def init_distributed(args):
    print(f"myrank: {args.rank} mygpu: {args.gpu}")
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        # if args.dist_url == "env://" and args.rank == -1:
        #     args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * args.ngpus_per_node + args.gpu
        init_process(backend=args.dist_backend,
                     init_method=args.dist_url,
                     size=args.world_size,
                     rank=args.rank)


def is_primary():
    return get_rank() == 0


def get_rank():
    if not dist.is_available():
        return 0

    if not dist.is_initialized():
        return 0

    return dist.get_rank()


LOCAL_PROCESS_GROUP = None


def get_local_rank():
    if not dist.is_available():
        return 0

    if not dist.is_initialized():
        return 0

    if LOCAL_PROCESS_GROUP is None:
        raise ValueError("tensorfn.distributed.LOCAL_PROCESS_GROUP is None")

    return dist.get_rank(group=LOCAL_PROCESS_GROUP)


def synchronize():
    if not dist.is_available():
        return

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()


def get_world_size():
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size()


def all_reduce(tensor, op=dist.ReduceOp.SUM):
    world_size = get_world_size()

    if world_size == 1:
        return tensor

    dist.all_reduce(tensor, op=op)

    return tensor