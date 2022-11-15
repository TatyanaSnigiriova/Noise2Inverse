#!/bin/bash -l
#export PATH=/opt/slurm/bin:$PATH
#SBATCH -J gen_rec # Job name
#SBATCH -p compclass # Queue name (another queue - compclass)
#SBATCH -o /home/t_kolesnik/noise2inverse_temp/phantom/split4_scripts/gen_rec_noise2inverse.%j.out #(%j expands to %jobId)
#SBATCH -e /home/t_kolesnik/noise2inverse_temp/phantom/split4_scripts/gen_rec_noise2inverse_error.%j.out #(%j expands to %jobId)
# #SBATCH -N 1 # Total number of nodes requested
# source ~/anaconda3/etc/profile.d/conda.sh
conda activate test_noise_env;

python 02_reconstruct.py \
    -cpu=0 -global_rs=42 \
    --reconstruction_library='noise2inverse' --reconstruction_method='fbp' \
    -sample_dir='phantom' \
    -data_dir='data' -logs_dir='logs' -proj_dir="projections" \
    --output_proj_prefix="output" -log_idx=50 \
    --center=-1 --rec_width=512 \
    -modes="circ_mask" -splits=4 \
    -norm=0 --circ_mask_ratio=0.9;
