#!/bin/bash -l
#export PATH=/opt/slurm/bin:$PATH
#SBATCH -J train_x_1 # Job name
#SBATCH -p gpuserv # Queue name (queues - gpuserv, compclass)
#SBATCH -o /home/t_kolesnik/noise2inverse_temp/sand_1p1x_31MP_034/split4_scripts/train_x_1.%j.out #(%j expands to %jobId)
#SBATCH -e /home/t_kolesnik/noise2inverse_temp/sand_1p1x_31MP_034/split4_scripts/train_x_1_error.%j.out #(%j expands to %jobId)
# #SBATCH -N 1 # Total number of nodes requested
# source ~/anaconda3/etc/profile.d/conda.sh
conda activate test_noise_env;

python 03_train.py \
    -cpu=0 -device=0 -global_rs=42 \
    --reconstruction_library='noise2inverse' --reconstruction_method='fbp' \
    -sample_dir='sand_1p1x_31MP_034' \
    -data_dir='data' -models_dir="models" -logs_dir='logs' \
    -prefix="output" -log_idx=50 \
    -modes="circ_mask" -splits=4 \
    -norm=1 -hist=99 \
    -strategy="X:1" -model="unet" -epochs=200 \
    -batch=4 --retrain_model=0;  ## >>instance3.out 2>&1 & # "1:X"