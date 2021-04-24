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

n_batch=256
h_dim=64
n_iter=40000
seed=0
modeltype=VQVAE
datatype=clevr
beta=1.0
i=0

ulimit -n 10000
ulimit -x unlimited

for n_latent in 8 16 32 64; do
  for n_codes in 16 24; do
    for lr in 0.0003; do
      i=$((i + 1));
      if [[ $i -eq $SLURM_ARRAY_TASK_ID ]]; then
        exp_folder="vis/${datatype}/${modeltype}/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}/logs"
        mkdir -p $exp_folder
        PYTHONHASHSEED=${seed} python -u main.py \
        --seed ${seed} \
        --n_batch ${n_batch} \
        --n_latent ${n_latent} \
        --n_codes ${n_codes} \
        --n_iter ${n_iter} \
        --h_dim ${h_dim} \
        --modeltype ${modeltype} \
        --datatype ${datatype} \
        --lr ${lr} > $exp_folder/eval.out 2> $exp_folder/eval.err
      fi
    done
  done
done
