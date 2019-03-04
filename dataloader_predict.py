# In[ ]:
# coding: utf-8

# ### 1. Import dependencies

# In[32]:


import numpy as np
# get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import torch.optim as optim
import os
from preprocess import makeFileNameAndStylePairs

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])

data_transforms = {
        'train':
        transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]),
        'validation':
        transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    normalize
                ]),
    }

image_datasets = {
        'train':
        datasets.ImageFolder('data/train', data_transforms['train']),
        'validation':
        datasets.ImageFolder('data/validation', data_transforms['validation'])
    }

dataloaders = {
        'train':
        torch.utils.data.DataLoader(image_datasets['train'],
                                                                    batch_size=32,
                                                                    shuffle=True, num_workers=4),
        'validation':
        torch.utils.data.DataLoader(image_datasets['validation'],
                                                                    batch_size=32,
                                                                    shuffle=False, num_workers=4)
    }


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=False).to(device)
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 21)).to(device)
model.load_state_dict(torch.load('models/pytorch/weights2.h5'))
model.eval()

running_corrects = 0
y_pred = []
y_true = []

for inputs, labels in dataloaders["validation"]:
    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    y_pred.extend(preds)
    y_true.extend(labels)
    running_corrects += torch.sum(preds == labels.data)
acc = running_corrects.float() / len(image_datasets["validation"])
print("accuracy: ", acc)
results = precision_recall_fscore_support(y_true, y_pred, average = "weighted")
print("precision and recall: ", results)
