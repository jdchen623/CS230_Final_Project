
# coding: utf-8

# ### 1. Import dependencies

# In[32]:


import numpy as np
# get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
from PIL import Image
import os


# In[33]:


import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim


# In[34]:


torch.__version__


# ### 2. Create PyTorch data generators

# In[35]:
WEIGHTS_FILE_NAME = 'testing'
WEIGHTS_DIR = "models/pytorch"
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, WEIGHTS_FILE_NAME)
NUM_FROZEN = 161 - 20
EPOCH_SAVE_FREQUENCY = 2
NUM_EPOCHS = 50
RESULTS_PATH = "results/testing_layers.txt"

try:
    os.mkdir(WEIGHTS_PATH)
except OSError:
    print ("Creation of the directory %s failed" % WEIGHTS_PATH)
else:
    print ("Successfully created the directory %s " % WEIGHTS_PATH)

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


# ### 3. Create the network

# In[36]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[37]:


model = models.resnet34(pretrained=True).to(device)

#layer_index = 0
#for param in model.paramters():
#    if layer_index < NUM_FROZEN:
#        param.requires_grad = False
#        layer_index += 1


count = 0
counts = []
for child in model.children():
    temp_count = 0
    for param in child.parameters():
        temp_count += 1
        param.requires_grad = False
        count += 1
    counts.append(temp_count)
print("This count is: "+ str(count))
print(counts)
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Dropout(0.2),
               nn.Linear(128, 10)).to(device)


# In[38]:


criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.fc.parameters())
optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()))

# ### 4. Train the model

# In[39]:

def train_model(model, criterion, optimizer, num_epochs=15):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.detach() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.float() / len(image_datasets[phase])

            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss.item(),
                                                        epoch_acc.item()))

            print('epoch: {}, {} loss: {:.4f}, acc: {:.4f}'.format(epoch+1, phase,
                                                        epoch_loss.item(),
                                                        epoch_acc.item()), file=open(RESULTS_PATH, "a"))
            if (epoch + 1) % EPOCH_SAVE_FREQUENCY == 0:
                torch.save(model.state_dict(), os.path.join(WEIGHTS_PATH, WEIGHTS_FILE_NAME + "_epoch" + str(epoch + 1) + ".h5"))

    return model


# In[ ]:


model_trained = train_model(model, criterion, optimizer, num_epochs=NUM_EPOCHS)


# ### 5. Save and load the model

# In[ ]:


torch.save(model_trained.state_dict(), WEIGHTS_PATH + ".h5")
