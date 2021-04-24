#!/bin/bash
#SBATCH --job-name=clevr
#SBATCH --time=58:00:00
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --qos=high
#SBATCH --constrain=xeon-g6
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-1


n_batch=256
h_dim=64
seed=0
modeltype=CVAE
datatype=clevr
i=0

ulimit -n 10000
ulimit -x unlimited

for n_latent in 8; do
  for beta in 1.0; do
    for lr in 0.0003; do
          i=$((i + 1));
          if [[ $i -eq $1 ]]; then
            exp_folder="vis/${datatype}/${modeltype}/beta_${beta}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}/logs/"
            vae_path="vis/${datatype}/VAE/beta_${beta}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}/model.pt"
            mkdir -p $exp_folder
            PYTHONHASHSEED=${seed} python -u main.py \
                                --seed ${seed} \
                                --n_batch ${n_batch} \
                      	        --n_latent ${n_latent} \
                             	  --h_dim ${h_dim} \
                                --beta ${beta} \
                                --modeltype ${modeltype} \
                                --datatype ${datatype} \
                                --rnn_dim 1600 \
                                --vae_path ${vae_path} \
                      	        --lr ${lr}
          fi
    done
  done
done
