#!/bin/bash -l

#$ -P ec523kb       # Specify the SCC project name you want to use
#$ -l h_rt=12:00:00   # Specify the hard time limit for the job
#$ -N myjob           # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -l gpus=1 #number of gpu
#$ -l gpu_c=4 #number of cores
#$ -pe omp 4 #reserve memory


module load miniconda
conda activate finalproject
python generate_low_res.py