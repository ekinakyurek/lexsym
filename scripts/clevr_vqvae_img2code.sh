#!/bin/bash
#SBATCH --job-name=clevr
#SBATCH --time=58:00:00
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --qos=high
#SBATCH --constrain=xeon-g6
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-16

n_batch=256
h_dim=64
seed=0
modeltype=VQVAE
datatype=clevr
i=0
beta=1.0

ulimit -n 10000
ulimit -x unlimited

for n_latent in 8 16 32 64; do
    for n_codes in 16 24; do
	     for lr in 0.0003; do
            i=$((i + 1));
            if [[ $i -eq $SLURM_ARRAY_TASK_ID ]]; then
          		exp_folder="vis/${datatype}/${modeltype}/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}/codes/logs"
              vae_path="vis/${datatype}/VQVAE/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}/model.pt"
          		mkdir -p $exp_folder
          		PYTHONHASHSEED=${seed} python -u main.py \
                                        --seed ${seed} \
                                        --n_batch ${n_batch} \
                                	      --n_latent ${n_latent} \
          			                        --n_codes ${n_codes} \
                                        --h_dim ${h_dim} \
                                        --modeltype ${modeltype} \
                                        --datatype ${datatype} \
                                        --extract_codes \
                                        --vae_path ${vae_path} \
                                	      --lr ${lr} > $exp_folder/eval.out 2> $exp_folder/eval.err
          fi
    done
  done
done
