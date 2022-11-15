#!/bin/bash -l
# source ~/anaconda3/etc/profile.d/conda.sh
conda create -n noise2inverse python=3.7
conda activate noise2inverse

#conda install msd_pytorch=0.7.2 cudatoolkit=10.2 -c aahendriksen -c pytorch -c defaults -c conda-forge -y
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch -y
conda install -c conda-forge cudatoolkit-dev -y
conda install -c conda-forge tomopy -y
conda install -c astra-toolbox/label/dev astra-toolbox -y

conda install -c conda-forge matplotlib -y
conda install -c conda-forge foam_ct_phantom -y
conda install -c anaconda h5py -y
conda install -c conda-forge tifffile -y
conda install -c anaconda pandas -y

#pip install git+https://github.com/ahendriksen/msd_pytorch.git@v0.7.2
pip install bm3d
pip install git+https://github.com/ahendriksen/tomosipo.git@v0.3.1
#conda deactivate
#conda env remove -n noise2inverse
