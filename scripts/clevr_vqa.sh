#!/bin/bash
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

n_batch=512
h_dim=64
seed=0
modeltype=VQA
datatype=clevr
i=0
vis_root='vis_vqa'
for n_codes in 32 ; do
  for n_latent in 64; do
    for beta in 1.0; do
      for lr in 0.0003; do
          i=$((i + 1))
          exp_root="${vis_root}/${datatype}/${modeltype}/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}"
      	  exp_folder="${exp_root}/codes/logs"
          vae_path="vis/${datatype}/VQVAE/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}/model.pt"
          code_root=${vae_path//vis/vis_test}
          code_root=${code_root//model.pt/}
      	  mkdir -p $exp_folder
      	  PYTHONHASHSEED=${seed} python -u vqa_train.py \
                                            --seed ${seed} \
                                            --n_batch ${n_batch} \
                                    	    --n_latent ${n_latent} \
              			                    --n_codes ${n_codes} \
                                            --h_dim ${h_dim} \
                                            --beta ${beta} \
                                            --modeltype ${modeltype} \
                                            --datatype ${datatype} \
                                            --vae_path ${vae_path} \
                                            --vis_root ${vis_root} \
                                            --n_workers 32 \
                                            --code_files "${code_root}/train_encodings.txt,${code_root}/test_encodings.txt" \
                                            --lr 1.0 \
                                            --warmup_steps 10000 \
                                            --visualize_every 1000
    done
  done
 done
done
