#!/bin/bash -l
#export PATH=/opt/slurm/bin:$PATH
#SBATCH -J gen_proj # Job name
#SBATCH -p compclass # Queue name (another queue - compclass)
#SBATCH -o /home/t_kolesnik/noise2inverse_temp/phantom/gen_proj.%j.out #(%j expands to %jobId)
#SBATCH -e /home/t_kolesnik/noise2inverse_temp/phantom/gen_proj_error.%j.out #(%j expands to %jobId)
# #SBATCH -N 1 # Total number of nodes requested
# source ~/anaconda3/etc/profile.d/conda.sh
conda activate test_noise_env;

python 01_generate_projections.py \
    -cpu=0 -global_r_s=42 \
    --working_dir='phantom' \
    --projs_path='phantom.h5' \
    --data_dir_name="data" --logs_dir_name="logs" \
    -prefix="output" --log_proj_idx=50;