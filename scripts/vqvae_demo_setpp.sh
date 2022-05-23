#!/bin/bash
n_batch=128
h_dim=32
seed=0
modeltype=VQVAE
datatype=setpp
i=0
vis_root='vis_test'
for n_codes in 8; do
  for n_latent in 32; do
    for beta in 1.0; do
      for lr in 0.0003; do
          i=$((i + 1));
          if [[ $i -eq $1 ]]; then
          exp_root="${vis_root}/${datatype}/${modeltype}/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}"
      	  exp_folder="${exp_root}/codes/logs"
          vae_path="${exp_root}/checkpoint.pth.tar"
          lex_path="${exp_root}/diag.align.json"
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
                                --resume ${vae_path} \
                                --lex_path ${lex_path} \
                                --vis_root ${vis_root} \
                       	        --lr ${lr}
          fi
      done
    done
  done
done
