#!/bin/bash
#SBATCH --job-name=clevr
#SBATCH --time=58:00:00
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=2
#SBATCH --qos=high
#SBATCH --constrain=xeon-g6
#SBATCH --gres=gpu:volta:2


source /etc/profile
module load cuda/11.1
module load mpi/openmpi-4.1.1
module load nccl/2.8.3-cuda11.1

export MPI_FLAGS="--tag-output --bind-to socket -map-by core -mca btl ^openib -mca pml ob1 -x PSM2_GPUDIRECT=1 -x NCCL_NET_GDR_LEVEL=5 -x NCCL_P2P_LEVEL=5 -x NCCL_NET_GDR_READ=1"

# Set some environment variables needed by torch.distributed 
export MASTER_ADDR=$(hostname -s)
# Get unused port
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "MASTER_ADDR : ${MASTER_ADDR}"
echo "MASTER_PORT : ${MASTER_PORT}"

# Do not use the torch.distributed.launch utility. Use mpirun as shown below
# to launch your code. The file torch_test.py has additional setup code needed to the
# distributed training capability 

n_batch=512
h_dim=128
seed=2
modeltype=VQVAE
datatype=clevr
i=0
beta=1.0
vis_root='vis_test_seed_2'
vqvae_root='vis_test_seed_2'
imgsize="128,128"
n_iter=100000

ulimit -n 10000
ulimit -x unlimited

eval "$(conda shell.bash hook)"
conda activate generative

LLSUB_RANK=${SLURM_PROCID}

for n_latent in 64; do
  for n_codes in 32; do
    for lr in 0.0003; do
        i=$((i + 1));
        # if [[ $i -eq $1 ]]; then
        exp_folder="${vis_root}/${datatype}/${modeltype}/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}/logs"
        mkdir -p $exp_folder

        PYTHONHASHSEED=${seed} mpirun ${MPI_FLAGS} python -u vae_train.py \
        --seed ${seed} \
        --n_batch ${n_batch} \
        --n_latent ${n_latent} \
        --n_codes ${n_codes} \
        --n_iter ${n_iter} \
        --h_dim ${h_dim} \
        --imgsize ${imgsize} \
        --modeltype ${modeltype} \
        --datatype ${datatype} \
        --lr ${lr} \
        --n_workers 16 \
        --vis_root ${vis_root} \
        --dataroot "data/clevr" \
        --world_size 4 \
        --dist_backend 'nccl' \
        --visualize_every 10000  > $exp_folder/eval.${LLSUB_RANK}.out 2> $exp_folder/eval.${LLSUB_RANK}.err
      # fi
    done
  done
done
