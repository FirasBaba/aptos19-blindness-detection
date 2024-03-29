from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import pretrainedmodels
import os


MODEL_NAME = os.environ["MODEL_NAME"]
TRAINING_BATCH_SIZE = int(os.environ["TRAINING_BATCH_SIZE"])
TEST_BATCH_SIZE = int(os.environ["TEST_BATCH_SIZE"])
IMAGE_SIZE = int(os.environ["IMAGE_SIZE"])
EPOCHS = int(os.environ["EPOCHS"])

device = torch.device("cuda")
model_ft = pretrainedmodels.__dict__[MODEL_NAME](pretrained='imagenet')


model_ft.avgpool_1a = nn.AdaptiveAvgPool2d(1)
model_ft.last_linear = nn.Sequential(
                          nn.BatchNorm1d(18432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                          nn.Dropout(p=0.25),
                          nn.Linear(in_features=18432, out_features=1024, bias=True),
                          nn.ReLU(),
                          nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                          nn.Dropout(p=0.5),
                          nn.Linear(in_features=1024, out_features=1, bias=True),
                         )
# model = nn.DataParallel(model_ft)
# model_ft = nn.DataParallel(model_ft, device_ids=[0,1])
model_ft = model_ft.to(device)



