#!/bin/bash
#SBATCH --job-name=clevr
#SBATCH --time=58:00:00
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --qos=high
#SBATCH --constrain=xeon-g6
#SBATCH --gres=gpu:volta:2

n_batch=512
h_dim=128
seed=0
modeltype=VQVAE
datatype=clevr
i=0
beta=1.0
imgsize="128,128"
n_workers=16
ulimit -n 10000
ulimit -x unlimited

eval "$(conda shell.bash hook)"
conda activate generative

for n_latent in 64; do
    for n_codes in 32; do
	     for lr in 0.0003; do
            vis_root="vis_test"
            exp_root="${vis_root}/${datatype}/${modeltype}/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}"
            vae_folder=$exp_root
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
                      --n_workers ${n_workers} \
                      --datatype ${datatype} \
                      --dataroot "data/clevr/" \
                      --resume ${vae_path} \
                      --vis_root ${vis_root} \
                      --imgsize ${imgsize} \
                      --lr ${lr} > $exp_folder/img2code.out 2> $exp_folder/img2code.err

            awk -F'\t' '{print $1" ||| "$2}' ${vae_folder}/train_encodings.txt  > ${vae_folder}/train_encodings.fast
            fast_align -i ${vae_folder}/train_encodings.fast -v > ${vae_folder}/forward.align
            fast_align -i ${vae_folder}/train_encodings.fast -v -r > ${vae_folder}/reverse.align
            fast_align -i ${vae_folder}/train_encodings.fast -o > ${vae_folder}/forward.align.o
            fast_align -i ${vae_folder}/train_encodings.fast -o -r > ${vae_folder}/reverse.align.o
            atools -i ${vae_folder}/forward.align -j ${vae_folder}/reverse.align -c intersect > ${vae_folder}/diag.align
            atools -i ${vae_folder}/forward.align -j ${vae_folder}/reverse.align -c grow-diag > ${vae_folder}/grow-diag.align
            atools -i ${vae_folder}/forward.align.o -j ${vae_folder}/reverse.align.o -c intersect > ${vae_folder}/diag.align.o
            atools -i ${vae_folder}/forward.align.o -j ${vae_folder}/reverse.align.o -c grow-diag > ${vae_folder}/grow-diag.align.o
            python seq2seq/utils/summarize_aligned_data.py ${vae_folder}/train_encodings.fast ${vae_folder}/forward.align
            python seq2seq/utils/summarize_aligned_data.py ${vae_folder}/train_encodings.fast ${vae_folder}/reverse.align
            python seq2seq/utils/summarize_aligned_data.py ${vae_folder}/train_encodings.fast ${vae_folder}/diag.align
            python seq2seq/utils/summarize_aligned_data.py ${vae_folder}/train_encodings.fast ${vae_folder}/grow-diag.align
            python seq2seq/utils/summarize_aligned_data.py ${vae_folder}/train_encodings.fast ${vae_folder}/forward.align.o
            python seq2seq/utils/summarize_aligned_data.py ${vae_folder}/train_encodings.fast ${vae_folder}/reverse.align.o
            python seq2seq/utils/summarize_aligned_data.py ${vae_folder}/train_encodings.fast ${vae_folder}/diag.align.o
            python seq2seq/utils/summarize_aligned_data.py ${vae_folder}/train_encodings.fast ${vae_folder}/grow-diag.align.o
            
            python lex_and_swaps.py --lexfile ${vae_folder}/diag.align.o.json --codefile ${vae_folder}/train_encodings.txt
            python lex_and_swaps.py --lexfile ${vae_folder}/diag.align.json --codefile ${vae_folder}/train_encodings.txt
    done
  done
done

