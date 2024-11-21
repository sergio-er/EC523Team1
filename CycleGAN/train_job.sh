#!/bin/bash -l


#$ -l h_rt=13:00:00

#$ -j y
#$ -m e
#$ -l gpus=2
#$ -l gpu_c=7
#$ -pe omp 4


module load miniconda

cd ~
cd ../../..
cd projectnb/ec523kb/projects/teams_Fall_2024/Team_1/pytorch-CycleGAN-and-pix2pix

conda activate pytorch-CycleGAN-and-pix2pix

python train.py --dataroot ./datasets/tinyface --name tinyface_cyclegan --model cycle_gan --display_id -1 --gpu_ids 0,1 --batch_size 8


