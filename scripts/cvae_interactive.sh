#!/bin/bash
n_batch=128
h_dim=32
seed=0
modeltype=CVAE
datatype=setpp
i=0
for n_latent in 16 32 64; do
  for beta in 1.0 5.0 10.0; do
    for lr in 0.001 0.0003; do
          i=$((i + 1));
          if [[ $i -eq $1 ]]; then
            exp_folder="vis/${datatype}/${modeltype}/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}/logs/"
            vae_path="vis/${datatype}/VAE/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}/model.pt"
            mkdir -p $exp_folder
            PYTHONHASHSEED=${seed} python -u main.py \
                                --seed ${seed} \
                                --n_batch ${n_batch} \
                      	        --n_latent ${n_latent} \
                             	  --h_dim ${h_dim} \
                                --beta ${beta} \
                                --modeltype ${modeltype} \
                                --datatype ${datatype} \
                                --rnn_dim 256 \
                                --vae_path ${vae_path} \
                      	        --lr ${lr}
          fi
    done
  done
done
