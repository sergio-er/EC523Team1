
DATAROOT="/projectnb/ec523kb/projects/teams_Fall_2024/Team_1/Bulat_datasets/HR"
CHECKPOINTS_DIR="/projectnb/ec523kb/projects/teams_Fall_2024/Team_1/nina/CycleGAN/checkpoints"
RESULTS_DIR="../results"
TEST_SCRIPT="../test.py"
MODEL_NAME="H2L_16_50k"

python $TEST_SCRIPT \
    --dataroot $DATAROOT \
    --name $MODEL_NAME \
    --model cycle_gan \
    --checkpoints_dir $CHECKPOINTS_DIR \
    --results_dir $RESULTS_DIR \
    --load_size 64 --crop_size 64 \
    --input_nc 3 --output_nc 3 \
    --direction AtoB
