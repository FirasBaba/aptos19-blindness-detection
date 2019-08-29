from __future__ import division
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import torch.nn as nn
import copy
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
from loss_metric import kappa_metric
from sklearn.metrics import mean_squared_error

def train_model(model, data_loader, dataset_sizes, device, optimizer, scheduler, num_epochs, fold_name, ):
    since = time.time()
    print("TRAIN based on best loss metric")
    criterion = nn.MSELoss()
    #criterion = FocalLoss()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 99999999
    all_scores = []
    best_score = -np.inf
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        valid_preds = np.zeros((dataset_sizes["val"]))
        valid_labels = np.zeros((dataset_sizes["val"]))
        val_bs = data_loader["val"].batch_size
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step(best_score)
                #scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
#             import pdb; pdb.set_trace()
            tk0 = tqdm(data_loader[phase], total=int(dataset_sizes[phase]/data_loader[phase].batch_size))
            counter = 0
            for bi, d in enumerate(tk0):                
                
                inputs = d["image"]
                labels = d["labels"].view(-1, 1)
                
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
#                     try:
                    outputs=model(inputs)
#                     except Exception as e: print('zah'); print(e)
                        
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                counter += 1
                tk0.set_postfix(loss=(running_loss/(counter * data_loader[phase].batch_size)))

                if phase == "val":
                    try:
                        valid_labels[bi * val_bs:(bi + 1) * val_bs] = labels.detach().cpu().squeeze().numpy()
                        valid_preds[bi * val_bs:(bi + 1) * val_bs] = outputs.detach().cpu().squeeze().numpy()
                    except :
                        print("error")
                        continue
               
                    

            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == "val":
                score = -mean_squared_error(valid_labels,valid_preds)
                all_scores.append(score)
                if score > best_score:
                    best_score = score
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), os.path.join(fold_name, "model.bin"))

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

#         if len(all_scores[-5:]) == 5:
#             if best_score not in all_scores[-5:]:
#                 break
#             if len(np.unique(all_scores)) == 1:
#                 break
#             if abs(min(all_scores[-5:]) - max(all_scores[-5:])) < 0.01:
#                 break
        print(all_scores[-20:])
#         import pdb; pdb.set_trace()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model