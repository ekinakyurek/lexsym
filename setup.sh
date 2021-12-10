#!/bin/bash
$SCRACTH=$HOME

git config --global user.name "Ekin Akyürek"
git config --global user.email "akyurekekin@gmail.com"

git clone git@github.com:ekinakyurek/lexgen.git
cd lexgen
git checkout bugfix

project_root=$(pwd)

if [ -d "$SCRACTH/anaconda3" ]
then
    cd $SCRACTH
    wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
    sh Anaconda3-2021.11-Linux-x86_64.sh -f -p $SCRACTH/anaconda3
    source ~/.bashrc
fi

cd $project_root
## Setup Conda Env
eval "$(conda shell.bash hook)"
conda create --name generativev2 python=3.8.5 numpy matplotlib imageio scipy notebook flake8
conda activate generativev2
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-nightly -c nvidia
pip install --no-cache-dir tensorboard tqdm h5py flask
conda clean

## Setup Clevr Data (18+24 GB)
curl https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip -o data/CLEVR_v1.0.zip
unzip data/CLEVR_v1.0.zip
curl https://dl.fbaipublicfiles.com/clevr/CLEVR_CoGenT_v1.0.zip -o data/CLEVR_CoGenT_v1.0.zip
unzip data/CLEVR_CoGenT_v1.0.zip


git clone git@github.com:clab/fast_align.git
cd fast_align
mkdir build
cd build
cmake ..
make

cd ${project_root}

cat "export PATH=\"${project_root}/fast_align/build:\$PATH\"" > ~/.bashrc




