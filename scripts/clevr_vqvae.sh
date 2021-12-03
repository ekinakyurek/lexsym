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
seed=0
modeltype=VQVAE
datatype=clevr
i=0
beta=1.0
vis_root='vis_clip'
vqvae_root='vis_clip'
CUDA_VISIBLE_DEVICES=0,1,2,3
imgsize="128,128"
n_iter=100000


ulimit -n 10000
ulimit -x unlimited

eval "$(conda shell.bash hook)"
conda activate generative

for n_latent in 64; do
  for n_codes in 32; do
    for lr in 0.0003; do
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
        --imgsize ${imgsize} \
        --modeltype ${modeltype} \
        --datatype ${datatype} \
        --lr ${lr} \
        --n_workers 32 \
        --vis_root ${vis_root} \
        --dataroot "data/clevr" \
        --visualize_every 10000 > $exp_folder/eval.out 2> $exp_folder/eval.err
      # fi
    done
  done
done
