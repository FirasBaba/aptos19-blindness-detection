export CUDA_VISIBLE_DEVICES=0,1
export MODEL_NAME="resnet50"
export TRAINING_BATCH_SIZE=32
export TEST_BATCH_SIZE=128
export IMAGE_SIZE=320
export EPOCHS=15

# python3 predict.py

python3 train_old.py --fold 0
# python3 train.py --fold 1
# python3 train.py --fold 2
# python3 train.py --fold 3
# python3  train.py --fold 4
