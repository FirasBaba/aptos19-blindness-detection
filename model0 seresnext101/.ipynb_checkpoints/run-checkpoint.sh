export CUDA_VISIBLE_DEVICES=0
export MODEL_NAME="se_resnext101_32x4d"
export TRAINING_BATCH_SIZE=16
export TEST_BATCH_SIZE=16
export IMAGE_SIZE=288
export EPOCHS=20

# python3 predict.py

python3 train.py --fold 0
# python3 train.py --fold 1
# python3 train.py --fold 2
# python3 train.py --fold 3
# python train.py --fold 4
