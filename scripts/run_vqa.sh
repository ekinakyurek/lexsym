#!/bin/bash

n_batch=1024
modeltype_vae=VQVAE
modeltype_vqa=VQA
datatype=clevr
n_workers=32
vqa_lr=1.0
beta=1.0
n_latent=64
n_codes=32
lr=0.0003
vae_iter=100000
vqa_iter=200000
h_dim=128
imgsize="128,128"
i=0

clevr_type=$1
aug=$2
seed=1

dataroot="data/${clevr_type}"
vae_root="new_vqvaes/seed_${seed}"
vqa_root="vqa_${aug}_seed_${seed}_${clevr_type}"

vqa_folder="${vqa_root}/${datatype}/${modeltype}/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}"
vae_folder="${vae_root}/${datatype}/VQVAE/beta_${beta}_ncodes_${n_codes}_ldim_${n_latent}_dim_${h_dim}_lr_${lr}"
vae_path="${vae_folder}/checkpoint.pth.tar"

mkdir -p $vqa_folder/logs

echo ${vqa_folder}
echo ${vae_folder}

lex_and_swaps_path=${vae_folder}/diag.align-swaps.json
code_root=${vae_folder}

params=("--seed=${seed}" \
"--n_batch=${n_batch}" \
"--n_latent=${n_latent}" \
"--n_codes=${n_codes}" \
"--h_dim=${h_dim}" \
"--beta=${beta}" \
"--n_iter=${vqa_iter}" \
"--modeltype=${modeltype_vqa}" \
"--datatype=${datatype}" \
"--vae_path=${vae_path}" \
"--vis_root=${vqa_root}" \
"--code_files=${code_root}/train_encodings.txt,${code_root}/test_encodings.txt,${code_root}/val_encodings.txt" \
"--imgsize=${imgsize}" \
"--n_workers=${n_workers}" \
"--lr=${vqa_lr}" \
"--gclip=5.0" \
"--rnn_dim=512" \
"--warmup_steps=16000" \
"--gaccum=1" \
"--dataroot=${dataroot}" \
"--visualize_every=5000")

[[ $aug == "swap" ]] && params+=("--lex_and_swaps_path=${lex_and_swaps_path}")
[[ $aug == "substitute" ]] && params+=("--lex_and_swaps_path=${lex_and_swaps_path}" "--substitute")


PYTHONHASHSEED=${seed} python -u vqa_train.py "${params[@]}"  > ${vqa_folder}/eval.err 2> ${vqa_folder}/eval.out
