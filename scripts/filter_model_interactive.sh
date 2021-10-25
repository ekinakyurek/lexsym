#!/bin/bash

#SBATCH -o filter.out-%A-%a
#SBATCH --qos=high
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH --tasks-per-node=1
#SBATCH --array=1-45

n_batch=364
n_epoch=100
seed=0
modeltype=FilterModel
datatype=clevr
lex_n_latent=128
lex_vae_type='VAE'
nsteps=8
i=0

for lr in 0.0005; do
 for beta in 1.0; do
    for n_latent in 16; do
	         h_dim=32   # 2*$n_latent
		     i=$((i + 1));
	         # if [[ $i -eq $1 ]]; then
          	     exp_folder="vis/${datatype}/${modeltype}/${lex_vae_type}/dim_${n_latent}_lr_${lr}_beta_${beta}"
          	     mkdir -p $exp_folder
          	     PYTHONHASHSEED=${seed} python -u main.py \
                                   --seed ${seed} \
                                   --n_batch ${n_batch} \
                        	       --n_latent ${n_latent} \
                                   --lex_n_latent ${lex_n_latent} \
                                   --lex_vae_type ${lex_vae_type} \
                                   --lex_text_conditional \
                                   --lex_n_steps ${nsteps} \
                                   --h_dim ${h_dim} \
                                   --n_epoch ${n_epoch} \
                                   --modeltype ${modeltype} \
                                   --datatype ${datatype} \
                                   --filter_model \
                                   --lr ${lr} \
                                   --n_workers 56 \
                                   --beta ${beta} \
                                   --vis_root exp_rnn_cond \
                                   --tensorboard  'logs'
		 # fi
       done
    done
done
