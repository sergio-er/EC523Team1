#!/bin/bash -l

#$ -P ec523kb                # Project assignment
#$ -pe omp 4                # Request 4 CPU cores
#$ -l gpus=2                # Request 2 GPUs
#$ -l gpu_c=7               # GPU compute capability
#$ -l gpu_memory=11G        # Request GPU memory
#$ -l h_rt=24:00:00         # Runtime (hh:mm:ss)
#$ -N cyclegan_train        # Job name
#$ -j y                     # Combine stdout and stderr
#$ -m ea                    # Email on end and abort
#$ -o logs/train_H2L_50k_cont.log  # Use $JOB_ID instead of %j

# Exit on any error
set -e

# Load required modules
module load miniconda
module load academic-ml
conda activate fall-2024-pyt

# Print job info
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Job ID: $JOB_ID"

# Navigate to project directory
cd /projectnb/ec523kb/projects/teams_Fall_2024/Team_1/nina/CycleGAN || exit 1

# Run training
python train.py \
    --dataroot ./datasets/H2L_50k \
    --name H2L_16_50k \
    --model cycle_gan \
    --display_id -1 \
    --gpu_ids 0,1 \
    --batch_size 8 \
    --save_epoch_freq 10 \
    --n_epochs 65 \
    --n_epochs_decay 110 \
    --netG simple_16 \
    --n_layers_D 0 \
    --display_freq 1000 \
    --load_size 16 \
    --continue_train \
    --epoch_count 45 \

# Print completion message
echo "Job completed at: $(date)"


