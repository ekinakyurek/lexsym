#!/bin/bash
#SBATCH --job-name=clevr
#SBATCH --time=64:00:00
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --qos=high
#SBATCH --constrain=xeon-g6
#SBATCH --gres=gpu:volta:2
#SBATCH --array=1-36

n_batch=512
h_dim=128
seed=0
modeltype_vae=VQVAE
modeltype_vqa=VQA
datatype=clevr
n_workers=16
vqa_lr=1.0
beta=1.0
vae_iter=100000
vqa_iter=200000

for imgsize in "128,128" "144,144"; do
    vqa_root="vis_vqa_${imgsize//,/\.}"
    vae_root="vis_img_${imgsize//,/\.}"
    for n_latent in 32 64 128; do
        for n_codes in 24 32 48; do
            for lr in 0.0003 0.0001; do
                i=$((i + 1))
                if [[ $i -eq $SLURM_ARRAY_TASK_ID ]]; then

                    vqa_folder="${vqa_root}/${datatype}/${modeltype}/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}"
                    vae_folder="${vae_root}/${datatype}/VQVAE/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}"
                    vae_path="${vae_folder}/checkpoint.pth.tar"
                    

                    mkdir -p $vqa_folder/logs
                    mkdir -p $vae_folder/logs

                    echo $vqa_folder
                    echo ${vae_folder}

                    PYTHONHASHSEED=${seed} python -u vae_train.py \
                    --seed ${seed} \
                    --n_batch ${n_batch} \
                    --n_latent ${n_latent} \
                    --n_codes ${n_codes} \
                    --n_iter ${vae_iter} \
                    --h_dim ${h_dim} \
                    --modeltype ${modeltype_vae} \
                    --datatype ${datatype} \
                    --lr ${lr} \
                    --n_workers ${n_workers} \
                    --vis_root ${vae_root} \
                    --imgsize ${imgsize} \
                    --visualize_every 1000 > $vae_folder/logs/eval.out 2> $vae_folder/logs/eval.err

                    PYTHONHASHSEED=${seed} python -u img2code.py \
                    --seed ${seed} \
                    --n_batch ${n_batch} \
                    --n_latent ${n_latent} \
                    --n_codes ${n_codes} \
                    --h_dim ${h_dim} \
                    --n_workers ${n_workers} \
                    --datatype ${datatype} \
                    --resume ${vae_path} \
                    --vis_root ${vae_root} \
                    --imgsize ${imgsize} \
                    --lr ${lr} > $vae_folder/logs/img2code.out 2> $vae_folder/logs/img2code.err

                    awk -F'\t' '{print $1" ||| "$2}' ${vae_folder}/train_encodings.txt  > ${vae_folder}/train_encodings.fast
                    fast_align -i ${vae_folder}/train_encodings.fast -v > ${vae_folder}/forward.align
                    fast_align -i ${vae_folder}/train_encodings.fast -v -r > ${vae_folder}/reverse.align
                    atools -i ${vae_folder}/forward.align -j ${vae_folder}/reverse.align -c intersect > ${vae_folder}/diag.align
                    atools -i ${vae_folder}/forward.align -j ${vae_folder}/reverse.align -c grow-diag > ${vae_folder}/grow-diag.align
                    python seq2seq/utils/summarize_aligned_data.py ${vae_folder}/train_encodings.fast ${vae_folder}/forward.align
                    python seq2seq/utils/summarize_aligned_data.py ${vae_folder}/train_encodings.fast ${vae_folder}/reverse.align
                    python seq2seq/utils/summarize_aligned_data.py ${vae_folder}/train_encodings.fast ${vae_folder}/diag.align
                    python seq2seq/utils/summarize_aligned_data.py ${vae_folder}/train_encodings.fast ${vae_folder}/grow-diag.align
                            
                    lex_path=${vae_folder}/diag.align.json

                    PYTHONHASHSEED=${seed} python -u vqa_train.py \
                    --seed ${seed} \
                    --n_batch ${n_batch} \
                    --n_latent ${n_latent} \
                    --n_codes ${n_codes} \
                    --h_dim ${h_dim} \
                    --beta ${beta} \
                    --n_iter ${vqa_iter} \
                    --modeltype ${modeltype_vqa} \
                    --datatype ${datatype} \
                    --vae_path ${vae_path} \
                    --lex_path ${lex_path} \
                    --vis_root ${vqa_root} \
                    --code_files "${vae_folder}/train_encodings.txt,${vae_folder}/test_encodings.txt" \
                    --imgsize ${imgsize} \
                    --n_workers ${n_workers} \
                    --lr ${vqa_lr} \
                    --warmup_steps 10000 \
                    --visualize_every 1000 > ${vqa_folder}/eval.err 2> ${vqa_folder}/eval.out
                fi
               done
          done
     done
done
