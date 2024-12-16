#!/bin/bash -l

#$ -P ec523kb
#$ -l gpus=1
#$ -l gpu_c=3.5
#$ -l h_rt=1:00:00
#$ -o logs/create_lr_$JOB_ID.log
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

# Navigate to ResShift directory
cd /projectnb/ec523kb/projects/teams_Fall_2024/Team_1/nina/ResShift

# Print detailed GPU information
nvidia-smi

# Run the LR pairs creation script with error handling
echo "\nStarting LR pairs creation..."
python create_cyclegan_lr.py || {
    echo "LR pairs creation failed with exit code $?"
    exit 1
}

echo "LR pairs creation completed successfully" 