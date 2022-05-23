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

n_batch=128
h_dim=64
seed=0
modeltype=VQVAE
datatype=setpp
i=0
vis_root='vis_test'
for n_codes in 32 16 ; do
  for n_latent in 64; do
    for beta in 0.1 3.0; do
      for lr in 0.0003; do
          i=$((i + 1))
          exp_root="${vis_root}/${datatype}/${modeltype}/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}"
      	  exp_folder="${exp_root}/codes/logs"
          vae_path="${exp_root}/checkpoint.pth.tar"
      	  mkdir -p $exp_folder
      	  PYTHONHASHSEED=${seed} python -u img2code.py \
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
                                            --vis_root ${vis_root} \
                                    	    --lr ${lr}
            awk -F'\t' '{print $1" ||| "$2}' ${exp_root}/train_encodings.txt  > ${exp_root}/train_encodings.fast
            fast_align -i ${exp_root}/train_encodings.fast -v > ${exp_root}/forward.align
            fast_align -i ${exp_root}/train_encodings.fast -v -r > ${exp_root}/reverse.align
            atools -i ${exp_root}/forward.align -j ${exp_root}/reverse.align -c intersect > ${exp_root}/diag.align
            atools -i ${exp_root}/forward.align -j ${exp_root}/reverse.align -c grow-diag > ${exp_root}/grow-diag.align
            python seq2seq/utils/summarize_aligned_data.py ${exp_root}/train_encodings.fast ${exp_root}/forward.align
            python seq2seq/utils/summarize_aligned_data.py ${exp_root}/train_encodings.fast ${exp_root}/reverse.align
            python seq2seq/utils/summarize_aligned_data.py ${exp_root}/train_encodings.fast ${exp_root}/diag.align
            python seq2seq/utils/summarize_aligned_data.py ${exp_root}/train_encodings.fast ${exp_root}/grow-diag.align
    done
  done
done
done
