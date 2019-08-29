export CUDA_VISIBLE_DEVICES=0
export MODEL_NAME="resnet50"
export TRAINING_BATCH_SIZE=32
export TEST_BATCH_SIZE=16
export IMAGE_SIZE=256
export EPOCHS=30

# python3 predict.py

python3 train.py --fold 0
# python3 train.py --fold 1
# python3 train.py --fold 2
# python3 train.py --fold 3
# python3  train.py --fold 4
