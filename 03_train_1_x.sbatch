#!/bin/bash -l
#export PATH=/opt/slurm/bin:$PATH
#SBATCH -J train_x_1 # Job name
#SBATCH -p compclass # Queue name (another queue - compclass)
#SBATCH -o /home/t_kolesnik/noise2inverse_temp/train_1_x.%j.out #(%j expands to %jobId)
#SBATCH -e /home/t_kolesnik/noise2inverse_temp/train_1_x_error.%j.out #(%j expands to %jobId)
# #SBATCH -N 1 # Total number of nodes requested
# source ~/anaconda3/etc/profile.d/conda.sh
conda activate noise2inverse-ssh;

python 03_train.py \
    -cpu=0 -device=0 -global_rs=42 \
    --reconstruction_library='noise2inverse' --reconstruction_method='fbp' \
    -sample_dir='sand_1p1x_31MP_034' \
    -data_dir='data' -models_dir="models" -logs_dir='logs' \
    -prefix="output" -log_idx=50 \
    -modes="circ_mask" \
    -strategy="1:X" -model="unet" -epochs=0 \
    -batch=4 --retrain_model=1 \
    -hist=99 -norm=1;  ## >>instance3.out 2>&1 & # "1:X"