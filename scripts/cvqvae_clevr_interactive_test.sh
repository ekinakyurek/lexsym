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


n_batch=128
h_dim=64
seed=0
modeltype=CVQVAE
datatype=clevr
beta=1.0
n_iter=100000
i=0

ulimit -n 10000
ulimit -x unlimited

for n_codes in 32; do
  for n_latent in 64; do
    for lr in 0.0003; do
          i=$((i + 1));
          if [[ $i -eq $1 ]]; then
            exp_folder="vis/${datatype}/${modeltype}/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}/"
            vae_path="vis/${datatype}/VQVAE/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}/model.pt"
            lex_path="vis/${datatype}/VQVAE/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}/diag.align.json"
            mkdir -p $exp_folder
            PYTHONHASHSEED=${seed} python -u main.py \
                                --seed ${seed} \
                                --n_batch ${n_batch} \
                      	        --n_latent ${n_latent} \
                             	  --h_dim ${h_dim} \
                                --n_codes ${n_codes} \
                                --modeltype ${modeltype} \
                                --datatype ${datatype} \
                                --rnn_dim 384 \
                                --vae_path ${vae_path} \
                                --model_path ${exp_folder}/model.pt \
                                --test \
                                --lex_path ${lex_path} \
                                --n_iter ${n_iter} \
                      	        --lr 0.0003
          fi
    done
  done
done
