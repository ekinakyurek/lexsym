## Setup Conda
project_root = $(pwd)

if [ -d "$SCRACTH/anaconda3" ]
then
    cd $SCRACTH
    wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-ppc64le.sh
    sh Anaconda3-2019.10-Linux-ppc64le.sh -f -p $SCRACTH/anaconda3
    source ~/.bashrc
fi

cd $project_root
## Setup Conda Env
conda env create -f environment.yml
eval "$(conda shell.bash hook)"
conda activate generative

## Setup Clevr Data
curl https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip -o data/CLEVR_v1.0.zip
unzip CLEVR_v1.0.zip
curl https://dl.fbaipublicfiles.com/clevr/CLEVR_CoGenT_v1.0.zip -o data/CLEVR_CoGenT_v1.0.zip
unzip CLEVR_CoGenT_v1.0.zip


git clone git@github.com:clab/fast_align.git
cd fast_align
mkdir build
cd build
cmake ..
make

cd ${project_root}

export PATH="${project_root}/fast_align/build:$PATH"


