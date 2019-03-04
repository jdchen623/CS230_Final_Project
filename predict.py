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

def get_validation_images():
    labels_dict = makeFileNameAndStylePairs()
    validation_file_names = []
    validation_file_labels = []
    validation_file_numeric_labels = []

    for root, dirs, files in os.walk("data/validation"):
        dir_index = 0
        labels = sorted(dirs)
        for directory in sorted(dirs):
            directory_path = os.path.join(root, directory)
            for subroot, subdir, subfiles in os.walk(directory_path):
                for subfile in subfiles:
                    if subfile == ".DS_Store": continue
                    file_path = os.path.join(subroot, subfile)
                    if subfile == ".DS_Store": continue
                    validation_file_names.append(os.path.join(subroot, subfile))
                    validation_file_labels.append(labels_dict[subfile])
                    validation_file_numeric_labels.append(dir_index)
            dir_index += 1
    return validation_file_names, validation_file_numeric_labels, labels

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



# ### 6. Make predictions on sample test images

# In[ ]:

validation_file_names, y_labels, label_names = get_validation_images()

validation_img_paths = validation_file_names

batch_size = 10
y_pred = []
img_list = [Image.open(img_path) for img_path in validation_img_paths]

num_images = len(img_list)

for i in range(0, int(len(img_list)/ batch_size)):
    img_batch = img_list[i * batch_size : (i + 1) * batch_size]
# In[ ]:
    validation_batch = torch.stack([data_transforms['validation'](img).to(device)
                                for img in img_batch])
    pred_logits_tensor = model(validation_batch)
    pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
    pred_index = np.argmax(pred_probs, axis = 1)
    y_pred.extend(pred_index)

if num_images % batch_size != 0:

    #edge case for total samples
    remaining_img_batch = len(img_list)%batch_size
    img_batch = img_list[len(img_list) - remaining_img_batch::]
    validation_batch = torch.stack([data_transforms['validation'](img).to(device)
                            for img in img_batch])
    pred_logits_tensor = model(validation_batch)
    pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
    pred_index = np.argmax(pred_probs, axis = 1)
    y_pred.extend(pred_index)
# In[ ]:

count = 0
for i in range(len(y_pred)):
    if y_pred[i] == y_labels[i]:
        count += 1

print("Count: " + str(count))
print("Percentage: " + str(float(count)/len(y_pred)))

results = precision_recall_fscore_support(y_labels, y_pred, average = "weighted")
conf_mat = confusion_matrix(y_labels, y_pred)
print(results)
print(results, file = open("results/precision_recall2.txt", 'w'))
print(conf_mat)
print(conf_mat, file = open("results/confusion_matrix.txt", 'w'))

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
array = conf_mat
df_cm = pd.DataFrame(array, index = [i for i in "ABCDEFGHIJKLMNOPQRSTU"],
                                       columns = [i for i in "ABCDEFGHIJKLMNOPQRSTU"])
plt.figure(figsize = (30, 30))
sn_plot = sn.heatmap(df_cm, annot=True)
fig = sn_plot.get_figure()
fig.savefig("results/output.png")




# In[ ]:


# fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
# for i, img in enumerate(img_list):
#     ax = axs[i]
#     ax.axis('off')
#     ax.set_title("{:.0f}% Alien, {:.0f}% Predator".format(100*pred_probs[i,0],
#                                                           100*pred_probs[i,1]))
#     ax.imshow(img)
