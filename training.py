import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import os
import torch.nn as nn
import torch.optim as optim


class ImageDataset(Dataset):
    def __init__(self, inputs, labels, transform=None, target_transform=None):
        self.img_labels = labels
        self.imgs = inputs
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

train = "./dataset/MineCraft-RT_1280x720_v14/MineCraft-RT_1280x720_v14/images"
print("Loading Training Data")
labels = []
inputs = []
for dirname, _, filenames in os.walk(train):
    for filename in filenames:
        label = cv2.imread(dirname +'/'+ filename)
        input = cv2.resize(label, (0,0), fx=0.25, fy=0.25) 
        labels.append(label)
        inputs.append(input)
        
train_dataset = ImageDataset(inputs,labels)    
        
test = "./labels/MineCraft-RT_1280x720_v12/MineCraft-RT_1280x720_v12"
print("Loading Test Data")
labels = []
inputs = []
for dirname, _, filenames in os.walk(test + '/images'):
    for filename in filenames:
        label = cv2.imread(dirname +'/'+ filename)
        input = cv2.resize(label, (0,0), fx=0.25, fy=0.25) 
        labels.append(label)
        inputs.append(input)
test_dataset = ImageDataset(inputs,labels)    

print(train_dataset.__getitem__(1)[0].shape,train_dataset.__getitem__(1)[1].shape)

train_dataloader = DataLoader(train_dataset, batch_size=4,shuffle=True, num_workers=0)
test_dataloader = DataLoader(train_dataset, batch_size=4,shuffle=True, num_workers=0)

###THIS IS UP TO THE POINT I ACTUALLY VERIFIED THINGS WORK
exit()

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        # Patch extraction and representation
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, padding=4)
        # Non-linear mapping
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)
        # Reconstruction
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, padding=2)
        # Activation function (ReLU)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

model = SRCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_dataloader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs.to(device)
        labels.to(device)
        
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.