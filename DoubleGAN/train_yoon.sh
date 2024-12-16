#!/bin/bash -l

#$ -P ec523kb       # Specify the SCC project name you want to use
#$ -l h_rt=24:00:00   # Specify the hard time limit for the job
#$ -N train_yoon           # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -l gpus=1 #number of gpu
#$ -l gpu_c=7 #number of cores
#$ -l gpu_memory=32G
#$ -pe omp 8 #reserve memory
#$ -o logs/train_$JOB_ID.log

echo "Starting job on host: $(hostname)"
echo "Date: $(date)"
echo "Directory: $(pwd)"

set -e

cd /projectnb/ec523kb/projects/teams_Fall_2024/Team_1/Sergio/BulatDoubleGan/unpaired_face_sr
module load miniconda
conda activate pytorch-CycleGAN-and-pix2pix

nvidia-smi
python yoon_train.py --gpu 0

echo "Training completed successfully"