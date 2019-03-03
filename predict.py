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
import torch.optim as optim
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=False).to(device)
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 33)).to(device)
model.load_state_dict(torch.load('models/pytorch/weights.h5'))


# ### 6. Make predictions on sample test images

# In[ ]:


validation_img_paths = ["data/validation/Cubism/1335.jpg"]
img_list = [Image.open(img_path) for img_path in validation_img_paths]


# In[ ]:


validation_batch = torch.stack([data_transforms['validation'](img).to(device)
                                for img in img_list])


# In[ ]:


pred_logits_tensor = model(validation_batch)
pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
print(pred_probs)

# In[ ]:


fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
for i, img in enumerate(img_list):
    ax = axs[i]
    ax.axis('off')
    ax.set_title("{:.0f}% Alien, {:.0f}% Predator".format(100*pred_probs[i,0],
                                                          100*pred_probs[i,1]))
    ax.imshow(img)



from preprocess import makeFileNameAndStylePairs

def get_validation_images():
    labels_dict = makeFileNameAndStylePairs()
    validation_file_names = []
    validation_file_labels = []

    for root, dirs, files in os.walk("data/validation"):
        for directory in dirs:
            directory_path = os.path.join(root, directory)

            for subroot, subdir, subfiles in os.walk(directory_path):
                for subfile in subfiles:
                    if subfile == ".DS_Store": continue
                    file_path = os.path.join(subroot, subfile)
                    img_label = labels_dict[subfile]
                    if type(img_label) is not str: continue
                    validation_file_names.append(os.path.join(subroot, subfile))
                    validation_file_labels.append(labels_dict[subfile])
    return validation_file_names, validation_file_labels
