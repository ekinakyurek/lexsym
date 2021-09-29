#!/bin/bash

#SBATCH -o filter.out-%A-%a
#SBATCH --qos=high
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH --tasks-per-node=1
#SBATCH --array=1-45

n_batch=64
n_epoch=50
seed=0
modeltype=FilterModel
datatype=clevr
i=0
for beta in  0.1 0.5 1.0 2.0 5.0; do
    for n_latent in 16 32 48; do
	     for lr in 0.00075 0.0005 0.00025; do
	         h_dim=$n_latent
		 i=$((i + 1));
	         if [[ $i -eq $SLURM_ARRAY_TASK_ID ]]; then
          	     exp_folder="vis/${datatype}/${modeltype}/dim_${n_latent}_lr_${lr}_beta_${beta}"
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
                                   --tensorboard  'tb_logs' 	      
		 fi
       done
    done
done
