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
        if ("left" in self.data.loc[idx, 'id_code']) or ("right" in self.data.loc[idx, 'id_code']) : 
            extension = ".jpeg"
            root = self.root_dir[1]
        else:
            extension = ".png"
            root = self.root_dir[0]
        img_name = os.path.join(root, self.data.loc[idx, 'id_code'] + extension)
        image = Image.open(img_name)
        labels = self.data.loc[idx, 'diagnosis'].astype(int)
        if self.transform:
            image = self.transform(image)
#         if ("left" in self.data.loc[idx, 'id_code']) or ("right" in self.data.loc[idx, 'id_code']) : 
#             print("left {}".format(image.shape))
#         else :
#             print("jdid {}".format(image.shape))

    
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
        self.data[self.data.image.apply( lambda x:x.split("_")[1])=="left"].reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.loc[idx, 'image'] + '.jpeg')
        image = Image.open(img_name)
        labels = self.data.loc[idx, 'level'].astype(int)
        if self.transform:
            image = self.transform(image)
    
        return {'image': image,
                'labels': labels
                }