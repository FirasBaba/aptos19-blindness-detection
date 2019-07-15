from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
import pretrainedmodels
import PIL
import os
import argparse
from PIL import Image

from APTOSdataset import APTOSDataset,APTOSOldDataset
from model import model_ft
from train_loop import train_model
from loss_metric import kappa_metric

print(kappa_metric)
parser = argparse.ArgumentParser()
parser.add_argument("--fold", default=-1)
args = parser.parse_args()

MODEL_NAME = os.environ["MODEL_NAME"]
TRAINING_BATCH_SIZE = int(os.environ["TRAINING_BATCH_SIZE"])
TEST_BATCH_SIZE = int(os.environ["TEST_BATCH_SIZE"])
IMAGE_SIZE = int(os.environ["IMAGE_SIZE"])
EPOCHS = int(os.environ["EPOCHS"])


FOLD = int(args.fold)
if FOLD == -1:
    FOLD = 0
training_folds = [i for i in range(5) if i!=FOLD]
val_folds = [FOLD]

    

FOLD_NAME = "fold{0}".format(FOLD)
if not os.path.exists(FOLD_NAME):
    os.makedirs(FOLD_NAME)
    

    
    
device = torch.device("cuda")
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD)
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD)
])

train_dataset = APTOSDataset(csv_file='../input/folds.csv',
                                   root_dir='../input/train_images/',
                                   image_size=IMAGE_SIZE,
                                   folds=training_folds,
                                   transform=train_transform)

valid_dataset = APTOSDataset(csv_file='../input/folds.csv',
                                   root_dir='../input/train_images/',
                                   image_size=IMAGE_SIZE,
                                   folds=val_folds,
                                   transform=val_transform)

old_train_dataset = APTOSOldDataset(csv_file='../input/2015_data/trainLabels.csv',
                                   root_dir='../input/2015_data/train/resized_train/',
                                   image_size=IMAGE_SIZE,
                                   transform=val_transform)



train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=TRAINING_BATCH_SIZE,
                                                   shuffle=True,
                                                   num_workers=1)

valid_dataset_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=TEST_BATCH_SIZE,
                                                   shuffle=False,
                                                   num_workers=1)

old_train_dataset_loader = torch.utils.data.DataLoader(old_train_dataset,
                                                   batch_size=TEST_BATCH_SIZE,
                                                   shuffle=False,
                                                   num_workers=1)

optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)
lr_sch = lr_scheduler.ReduceLROnPlateau(optimizer_ft, verbose=True, factor=0.2, mode="max", patience=2, threshold=0.01)

dataset_sizes = {}
dataset_sizes["train"] = len(train_dataset)
# dataset_sizes["val"] = len(valid_dataset)
dataset_sizes["val"] = len(old_train_dataset)

data_loader = {}
data_loader["train"] = train_dataset_loader
# data_loader["val"] = valid_dataset_loader
data_loader["val"] = old_train_dataset_loader

FOLD_NAME = "fold{0}".format(FOLD)
model_ft = train_model(model_ft,
                       data_loader,
                       dataset_sizes,
                       device,
                       optimizer_ft,
                       lr_sch,
                       num_epochs=EPOCHS,
                       fold_name=FOLD_NAME)

torch.save(model_ft.state_dict(), os.path.join(FOLD_NAME, "model.bin"))





