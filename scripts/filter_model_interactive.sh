#!/bin/bash
n_batch=64
n_latent=64
n_epoch=50
seed=0
modeltype=FilterModel
datatype=setpp
i=0
for lr in 0.001 0.0003; do
  i=$((i + 1));
  if [[ $i -eq $1 ]]; then
    exp_folder="vis/${datatype}/${modeltype}/dim_${n_latent}_lr_${lr}/logs/"
    mkdir -p $exp_folder
    PYTHONHASHSEED=${seed} python -u main.py \
                        --seed ${seed} \
                        --n_batch ${n_batch} \
              	        --n_latent ${n_latent} \
                        --n_epoch ${n_epoch} \
                        --modeltype ${modeltype} \
                        --datatype ${datatype} \
                        --filter_model \
                        --lr ${lr} \
                        --n_workers 4 \
                        # --gpu 0 \
                        # --distributed \
                        # --multiprocessing_distributed \
                        # --rank 0 \
                        # --world_size 1 \

  fi
done
