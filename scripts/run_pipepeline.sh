#!/bin/bash
#SBATCH --job-name=clevr
#SBATCH --time=64:00:00
#SBATCH --cpus-per-task=20
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --qos=high
#SBATCH --constrain=xeon-g6
#SBATCH --gres=gpu:volta:2
#SBATCH --array=1-8

n_batch=512
modeltype_vae=VQVAE
modeltype_vqa=VQA
datatype=clevr
n_workers=20
vqa_lr=1.0
beta=1.0
vae_iter=100000
vqa_iter=200000
h_dim=128
imgsize="128,128"
i=0

for clevr_type in "clevr" "original_clevr"; do
    for seed in 0 1 2 3; do
        dataroot="data/${clevr_type}"
        vqa_root="clip_10_exp_vqa_seed_${seed}_${clevr_type}"
        vae_root="clip_10_exp_img_seed_${seed}_${clevr_type}"
        for n_latent in 64; do
            for n_codes in 32; do
                for lr in 0.0003; do
                    i=$((i + 1))
                    if [[ $i -eq $SLURM_ARRAY_TASK_ID ]]; then

                        vqa_folder="${vqa_root}/${datatype}/${modeltype}/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}"
                        vae_folder="${vae_root}/${datatype}/VQVAE/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}"
                        vae_path="${vae_folder}/checkpoint.pth.tar"
                        

                        mkdir -p $vqa_folder/logs
                        mkdir -p $vae_folder/logs

                        echo ${vqa_folder}
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
                        --gaccum 1 \
			--gclip 10.0 \
                        --n_workers ${n_workers} \
                        --vis_root ${vae_root} \
                        --imgsize ${imgsize} \
                        --dataroot ${dataroot} \
                        --visualize_every 10000 > $vae_folder/logs/eval.out 2> $vae_folder/logs/eval.err

                        PYTHONHASHSEED=${seed} python -u img2code.py \
                        --seed ${seed} \
                        --n_batch 512 \
                        --n_latent ${n_latent} \
                        --n_codes ${n_codes} \
                        --h_dim ${h_dim} \
                        --n_workers ${n_workers} \
                        --datatype ${datatype} \
                        --dataroot ${dataroot} \
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

                        # PYTHONHASHSEED=${seed} python -u vqa_train.py \
                        # --seed ${seed} \
                        # --n_batch ${n_batch} \
                        # --n_latent ${n_latent} \
                        # --n_codes ${n_codes} \
                        # --h_dim ${h_dim} \
                        # --beta ${beta} \
                        # --n_iter ${vqa_iter} \
                        # --modeltype ${modeltype_vqa} \
                        # --datatype ${datatype} \
                        # --vae_path ${vae_path} \
                        # --lex_path ${lex_path} \
                        # --vis_root ${vqa_root} \
                        # --code_files "${code_root}/train_encodings.txt,${code_root}/test_encodings.txt,${code_root}/val_encodings.txt" \
                        # --imgsize ${imgsize} \
                        # --n_workers ${n_workers} \
                        # --lr ${vqa_lr} \
                        # --gclip 5.0 \
                        # --warmup_steps 10000 \
                        # --gaccum 2 \
                        # --dataroot ${dataroot} \
                        # --visualize_every 1000 > ${vqa_folder}/eval.err 2> ${vqa_folder}/eval.out
                    fi
                done
            done
        done
    done
done
