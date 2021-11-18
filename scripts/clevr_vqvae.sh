#!/bin/bash
#SBATCH --job-name=clevr
#SBATCH --time=58:00:00
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --qos=high
#SBATCH --constrain=xeon-g6
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-8

n_batch=512
h_dim=128
n_iter=100000
seed=0
modeltype=VQVAE
datatype=clevr
beta=1.0
i=0

ulimit -n 10000
ulimit -x unlimited

CUDA_VISIBLE_DEVICES=0,1,2,3
vis_root="visv2"
for n_latent in 64 128; do
  for n_codes in 32 48; do
    for lr in 0.0003 0.0005; do
      i=$((i + 1));
      # if [[ $i -eq $1 ]]; then
        exp_folder="${vis_root}/${datatype}/${modeltype}/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}/logs"
        mkdir -p $exp_folder
        PYTHONHASHSEED=${seed} python -u vae_train.py \
        --seed ${seed} \
        --n_batch ${n_batch} \
        --n_latent ${n_latent} \
        --n_codes ${n_codes} \
        --n_iter ${n_iter} \
        --h_dim ${h_dim} \
        --modeltype ${modeltype} \
        --datatype ${datatype} \
        --lr ${lr} \
        --n_workers 16 \
        --vis_root ${vis_root} \
        --visualize_every 1000 > $exp_folder/eval.out 2> $exp_folder/eval.err
      # fi
    done
  done
done
