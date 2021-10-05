#!/bin/bash

#SBATCH -o filter.out-%A-%a
#SBATCH --qos=high
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH --tasks-per-node=1
#SBATCH --array=1-45

n_batch=512
n_epoch=50
seed=0
modeltype=FilterModel
datatype=clevr
i=0
lex_n_latent=64
lex_vae_type='VQVAE'
ncodes=4
for beta in 1.0 2.0 5.0; do
    for n_latent in 4 16 32 48; do
	     for lr in 0.0003 0.0005 0.00025; do
	         h_dim=64  #  $n_latent
		     i=$((i + 1));
	         if [[ $i -eq $1 ]]; then
          	     exp_folder="vis/${datatype}/${modeltype}/${lex_vae_type}/dim_${n_latent}_lr_${lr}_beta_${beta}"
          	     mkdir -p $exp_folder
          	     PYTHONHASHSEED=${seed} python -u main.py \
                                   --seed ${seed} \
                                   --n_batch ${n_batch} \
                        	       --n_latent ${n_latent} \
                                   --n_codes ${ncodes} \
                                   --lex_n_latent ${lex_n_latent} \
                                   --lex_vae_type ${lex_vae_type} \
                                   --h_dim ${h_dim} \
                                   --n_epoch ${n_epoch} \
                                   --modeltype ${modeltype} \
                                   --datatype ${datatype} \
                                   --filter_model \
                                   --lr ${lr} \
                                   --n_workers 12 \
                                   --beta ${beta} \
                                   --tensorboard  'tb_logs'
		 fi
       done
    done
done
