#!/bin/bash
#SBATCH --job-name=simplecolor
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --qos=high
#SBATCH --constrain=xeon-g6
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-54
n_batch=128
h_dim=32
seed=0
i=0
for n_codes in 12 24 32; do
  for n_latent in 16 32 64; do
    for beta in 1.0 5.0 10.0; do
      for lr in 0.001 0.0003; do
          i=$((i + 1));
          if [[ $i -eq $SLURM_ARRAY_TASK_ID ]]; then
            PYTHONHASHSEED=${seed} python main.py \
                                --seed ${seed} \
                                --n_batch ${n_batch} \
                      	        --n_latent ${n_latent} \
                                --n_codes ${n_codes}\
                             	  --h_dim ${dim} \
                      	        --lr ${lr} > eval.$i.out 2> eval.$i.err
          fi
      done
    done
  done
done
