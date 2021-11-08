#!/bin/bash
n_batch=128
h_dim=64
seed=0
modeltype=VQVAE
datatype=setpp
i=0
for n_codes in 32 16 ; do
  for n_latent in 64; do
    for beta in 0.1 3.0; do
      for lr in 0.0003; do
          i=$((i + 1));
          # if [[ $i -eq $1 ]]; then
            exp_folder="vis_test/${datatype}/${modeltype}/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}/logs/"
            mkdir -p $exp_folder
            echo 'logging to'
            echo ${exp_folder}
            PYTHONHASHSEED=${seed} python -u vae_train.py \
                                            --seed ${seed} \
                                            --n_batch ${n_batch} \
                                  	        --n_latent ${n_latent} \
                                            --n_codes ${n_codes} \
                                         	--h_dim ${h_dim} \
                                            --beta ${beta} \
                                            --modeltype ${modeltype} \
                                            --datatype ${datatype} \
                                   	        --lr ${lr} \
                                            --visualize_every 1000 \
                                            --vis_root vis_test > $exp_folder/eval.out 2> $exp_folder/eval.err
          # fi
      done
    done
  done
done
