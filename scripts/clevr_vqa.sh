#!/bin/bash
#SBATCH --job-name=clevr
#SBATCH --time=58:00:00
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=4
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

ulimit -n 10000
ulimit -x unlim```````````````````````````````````````````````````````````````````````                                    1```````````````````````````````````````````````````````````````````````d


# Do not use the torch.distributed.launch utility. Use mpirun as shown below
# to launch your code. The file torch_test.py has additional setup code needed to the
# distributed training capability 

n_batch=1024
gaccum=1
h_dim=128
seed=0
modeltype=VQA
datatype=clevr
i=0
beta=1.0
vis_root='vqa_test_seed_2'
vqvae_root='vqa_exp_folder/clip_5_folders/clip_exp_img_seed_2_clevr'
imgsize="128,128"
vqa_lr=1.5
warmup_steps=30000
eval "$(conda shell.bash hook)"
conda activate generative


LLSUB_RANK=${SLURM_PROCID}

# rnn_dim 386
for n_latent in 64; do
  for n_codes in 32; do
    for lr in 0.0003; do
          i=$((i + 1))
          exp_root="${vis_root}/${datatype}/${modeltype}/beta_${beta}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}"
      	  exp_folder="${exp_root}/logs"
          vae_path="${vqvae_root}/${datatype}/VQVAE/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}/checkpoint.pth.tar"
          code_root=${vae_path//checkpoint.pth.tar/}
          lex_and_swaps_path="${vqvae_root}/${datatype}/VQVAE/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}/diag.align-swaps.json"
      	  mkdir -p $exp_folder
      	  PYTHONHASHSEED=0 mpirun ${MPI_FLAGS} python -u vqa_train.py \
                                            --seed ${seed} \
                                            --n_batch ${n_batch} \
                                    	      --n_latent ${n_latent} \
                			                      --n_codes ${n_codes} \
                                            --h_dim ${h_dim} \
                                            --beta ${beta} \
                                            --modeltype ${modeltype} \
                                            --datatype ${datatype} \
                                            --vae_path ${vae_path} \
                                            --vis_root ${vis_root} \
                                            --rnn_dim 512 \
                                            --n_workers 16 \
                                            --imgsize ${imgsize} \
                                            --code_files "${code_root}/train_encodings.txt,${code_root}/test_encodings.txt,${code_root}/val_encodings.txt" \
                              					    --lex_and_swaps_path ${lex_and_swaps_path} \
                                            --lr 1.0 \
                                            --gclip 5.0 \
                                            --gaccum ${gaccum} \
                                            --warmup_steps ${warmup_steps} \
                                            --dataroot "data/clevr/" \
                                            --world_size 8 \
                                            --dist_backend 'nccl' \
                                            --visualize_every 10000 > $exp_folder/eval.${LLSUB_RANK}.out 2> $exp_folder/eval.${LLSUB_RANK}.err
  done 
 done
done
