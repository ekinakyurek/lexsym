#!/bin/bash
n_batch=128
h_dim=64
seed=0
modeltype=VQVAE
datatype=clevr
i=0
for n_codes in 32; do
  for n_latent in 64; do
    for beta in 1.0; do
      for lr in 0.0003; do
          i=$((i + 1));
          if [[ $i -eq $1 ]]; then
            exp_folder="vis/${datatype}/${modeltype}/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}/logs"
            vae_path="vis/${datatype}/${modeltype}/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}/model.pt"
            lex_path="vis/${datatype}/${modeltype}/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}/diag.align.json"
            mkdir -p $exp_folder
            PYTHONHASHSEED=${seed} python -u demo.py \
                                --seed ${seed} \
                                --n_batch ${n_batch} \
                      	        --n_latent ${n_latent} \
                                --n_codes ${n_codes} \
                             	--h_dim ${h_dim} \
                                --beta ${beta} \
                                --modeltype ${modeltype} \
                                --datatype ${datatype} \
                                --vae_path ${vae_path} \
                                --lex_path ${lex_path} \
                       	        --lr ${lr}
          fi
      done
    done
  done
done