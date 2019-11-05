from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import datasets, models, transforms
import os
from tqdm import tqdm
import copy
import pretrainedmodels
import joblib
//bien
from APTOSdataset import APTOSOldDataset,APTOSDataset
from model import model_ft
from train_loop import train_model
from loss_metric import kappa_metric


MODEL_NAME = os.environ["MODEL_NAME"]
TRAINING_BATCH_SIZE = int(os.environ["TRAINING_BATCH_SIZE"])
TEST_BATCH_SIZE = int(os.environ["TEST_BATCH_SIZE"])
IMAGE_SIZE = int(os.environ["IMAGE_SIZE"])

FOLD = 3
if FOLD == -1:
    FOLD = 0
training_folds = [i for i in range(5) if i!=FOLD]
val_folds = [FOLD]

device = torch.device("cuda:0")
IMG_MEAN = model_ft.mean
IMG_STD = model_ft.std


train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD)
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD)
])

valid_dataset = APTOSDataset(csv_file='../input/folds.csv',
                                   root_dir='../input/train_images/',
                                   image_size=IMAGE_SIZE,
                                   folds=val_folds,
                                   transform=val_transform)


valid_dataset_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=TEST_BATCH_SIZE,
                                                   shuffle=False,
                                                   num_workers=2)

# valid_dataset = APTOSOldDataset(csv_file='../input/2015_data/trainLabels.csv',
#                                    root_dir='../input/2015_data/train/resized_train/',
#                                    image_size=IMAGE_SIZE,
#                                    transform=val_transform)


# valid_dataset_loader = torch.utils.data.DataLoader(valid_dataset,
#                                                    batch_size=TEST_BATCH_SIZE,
#                                                    shuffle=False,
#                                                    num_workers=1)
FOLD_NAME = "fold{0}".format(0)

model_ft.load_state_dict(torch.load(os.path.join(FOLD_NAME, "model.bin")))
model_ft = model_ft.to(device)

for param in model_ft.parameters():
    param.requires_grad = False
    
    
model_ft.eval()
valid_preds = np.zeros((len(valid_dataset)))
valid_labels = np.zeros((len(valid_dataset)))
tk0 = tqdm(valid_dataset_loader)
for i, _batch in enumerate(tk0):
    x_batch = _batch["image"]
    y_batch = _batch["labels"]
    pred = model_ft(x_batch.to(device))
    valid_labels[i * TEST_BATCH_SIZE:(i + 1) * TEST_BATCH_SIZE] = y_batch.detach().cpu().squeeze().numpy()
    valid_preds[i * TEST_BATCH_SIZE:(i + 1) * TEST_BATCH_SIZE] = pred.detach().cpu().squeeze().numpy()
    
print("The final Kappa score on the old dataset is : " , kappa_metric(valid_labels,valid_preds))
