#!/bin/bash -l

#$ -P ec523kb
#$ -l gpus=4            # Request 4 GPUs
#$ -l gpu_c=7.0
#$ -l gpu_memory=32G
#$ -pe omp 8
#$ -l h_rt=24:00:00
#$ -o logs/train_$JOB_ID.log
#$ -j y

# Print job information
echo "Starting job on host: $(hostname)"
echo "Date: $(date)"
echo "Directory: $(pwd)"

# Fail on any error
set -e

# Load required modules
module load miniconda
module load academic-ml
module load cuda
module load gcc/12.2.0

# Activate conda environment
conda activate fall-2024-pyt

# Navigate to your ResShift directory
cd /projectnb/ec523kb/projects/teams_Fall_2024/Team_1/nina/ResShift_CycleGAN

# Print detailed GPU information
nvidia-smi

# Run the training with distributed setup for 4 GPUs
echo "\nStarting training with 4 GPUs..."
torchrun \
    --standalone \
    --nproc_per_node=4 \
    --nnodes=1 \
    main.py \
    --cfg_path configs/realsr_swinunet_realesrgan256_train.yaml \
    --save_dir logs/data_16x16 || {
    echo "Training failed with exit code $?"
    exit 1
}

echo "Training completed successfully"