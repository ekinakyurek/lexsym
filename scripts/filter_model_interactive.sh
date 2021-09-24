#!/bin/bash
n_batch=256
n_latent=64
h_dim=64
n_epoch=50
beta=5
seed=0
modeltype=FilterModel
datatype=clevr
i=0
for lr in 0.001; do
  i=$((i + 1));
  if [[ $i -eq $1 ]]; then
    exp_folder="vis/${datatype}/${modeltype}/dim_${n_latent}_lr_${lr}/logs/"
    mkdir -p $exp_folder
    PYTHONHASHSEED=${seed} python -u main.py \
                        --seed ${seed} \
                        --n_batch ${n_batch} \
              	        --n_latent ${n_latent} \
                        --h_dim ${h_dim} \
                        --n_epoch ${n_epoch} \
                        --modeltype ${modeltype} \
                        --datatype ${datatype} \
                        --filter_model \
                        --lr ${lr} \
                        --n_workers 8 \
                        --beta ${beta} \
                        --tensorboard 'tb_logs'\
                        # --gpu 0 \
                        # --distributed \
                        # --multiprocessing_distributed \
                        # --rank 0 \
                        # --world_size 1 \

  fi
done
