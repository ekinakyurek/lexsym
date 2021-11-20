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
h_dim=128
seed=0
modeltype=VQVAE
datatype=clevr
i=0
beta=1.0


ulimit -n 10000
ulimit -x unlimited

CUDA_VISIBLE_DEVICES=12,13,14,15

eval "$(conda shell.bash hook)"
conda activate generative
vis_root='visv2'

for n_latent in 64 128; do
  for n_codes in 32 48; do
    for lr in 0.0003 0.0005; do
             exp_root="${vis_root}/${datatype}/${modeltype}/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}"
             exp_folder="${vis_root}/${datatype}/${modeltype}/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}/codes/logs"
             vae_path="${vis_root}/${datatype}/VQVAE/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}/checkpoint.pth.tar"
             if [[ -f $vae_path ]]; then
              mkdir -p $exp_folder
              PYTHONHASHSEED=${seed} python -u img2code.py \
                                                --seed ${seed} \
                                                --n_batch ${n_batch} \
                                                --n_latent ${n_latent} \
                                                --n_codes ${n_codes} \
                                                --h_dim ${h_dim} \
                                                --n_workers 16 \
                                                --datatype ${datatype} \
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
            fi
    done
  done
done
