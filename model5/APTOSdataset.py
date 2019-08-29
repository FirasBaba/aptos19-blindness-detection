from __future__ import division
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import os
import logging
import cv2


class APTOSDataset(Dataset):
    def __init__(self, csv_file, root_dir, image_size, folds=None, transform=None):
        if folds is None:
            folds = []
        self.data = pd.read_csv(csv_file)
        if len(folds) > 0:
            self.data = self.data[self.data.fold.isin(folds)].reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.loc[idx, 'id_code'] + '.png')
#         image = Image.open(img_name)
        image = cv2.imread(img_name)
        labels = self.data.loc[idx, 'diagnosis'].astype(int)
        if self.transform:
            image = self.transform(image=image)
            image = image['image']
    
        return {'image': image,
                'labels': labels
                }

class APTOSDatasetTest(Dataset):
    def __init__(self, csv_file, root_dir, image_size, folds=None, transform=None):
        if folds is None:
            folds = []
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.loc[idx, 'id_code'] + '.png')
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
    
        return {'image': image}
    
class APTOSOldDataset(Dataset):
    def __init__(self, csv_file, root_dir, image_size, transform=None):
        self.data = pd.read_csv(csv_file)
#         self.data[self.data.image.apply( lambda x:x.split("_")[1])=="left"].reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.loc[idx, 'image'] + '.jpeg')
        
        image = cv2.imread(img_name)
        labels = self.data.loc[idx, 'level'].astype(int)
        if self.transform:
            image = self.transform(image=image)
            image = image['image']
        

    
        return {'image': image,
                'labels': labels
                }