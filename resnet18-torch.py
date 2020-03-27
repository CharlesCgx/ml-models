#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Pytorch ResNet-18 ###

import torch
from torch.nn import Conv2d, MaxPool2d
import torch.nn as nn
import torch.nn.functional as F
import time
from keras.preprocessing.image import ImageDataGenerator

# ResNet Building blocks

class ResidualBlock(nn.Module): 
    def __init__(self, inchannel, outchannel, stride=1): 
        
        super(ResidualBlock, self).__init__() 
        
        self.left = nn.Sequential(Conv2d(inchannel, outchannel, kernel_size=3, 
                                         stride=stride, padding=1, bias=False), 
                                  nn.BatchNorm2d(outchannel), 
                                  nn.ReLU(inplace=True), 
                                  Conv2d(outchannel, outchannel, kernel_size=3, 
                                         stride=1, padding=1, bias=False), 
                                  nn.BatchNorm2d(outchannel)) 
        self.shortcut = nn.Sequential() 
        if stride != 1 or inchannel != outchannel: 

            self.shortcut = nn.Sequential(Conv2d(inchannel, outchannel, 
                                                 kernel_size=1, stride=stride, 
                                                 padding = 0, bias=False), 
                                          nn.BatchNorm2d(outchannel) ) 
            
    def forward(self, x): 
        
        out = self.left(x) 
        out += self.shortcut(x) 
        out = F.relu(out) 
        
        return out


    
# ResNet

class ResNet(nn.Module):
    
    def __init__(self, ResidualBlock, num_classes = 10):
        
        super(ResNet, self).__init__()
        
        self.inchannel = 64
        self.conv1 = nn.Sequential(Conv2d(3, 64, kernel_size = 3, stride = 1,
                                            padding = 1, bias = False), 
                                  nn.BatchNorm2d(64), 
                                  nn.ReLU())
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride = 1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride = 2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride = 2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride = 2)
        self.maxpool = MaxPool2d(4)
        self.fc = nn.Linear(512, num_classes)
        
    
    def make_layer(self, block, channels, num_blocks, stride):
        
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides: 
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
            
        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    
def ResNet18():
    return ResNet(ResidualBlock)


### Example with CIFAR-10 dataset ###

def check_accuracy(loader, model):
    """Computes test accuracy on validation and test set"""
    
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

        
def train_part(model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        print(len(loader_train))
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = torch.Tensor(x).to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = torch.Tensor(y).to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            loss.backward()

            # Update the parameters of the model using the gradients
            optimizer.step()

            if t % print_every == 0:
                print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                #check_accuracy(loader_val, model)
                print()
            if t > len(loader_train):
              break
            
USE_GPU = True
dtype = torch.float32 
print_every = 100

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')    
    
#### Load and split data ####
data_dir = './data'

n_img = 50000

# Training set
data_train = dset.CIFAR10(data_dir, train=True, download=True, transform=transform)
loader_train_0 = DataLoader(data_train, batch_size=n_img, sampler=sampler.SubsetRandomSampler(range(n_img)))
# Validation set
data_val = dset.CIFAR10(data_dir, train=True, download=True, transform=transform)
loader_val = DataLoader(data_val, batch_size=int(n_img*0.9), sampler=sampler.SubsetRandomSampler(range(int(n_img*0.90), n_img)))
# Testing set
data_test = dset.CIFAR10(data_dir, train=False, download=True, transform=transform)
loader_test = DataLoader(data_test, batch_size=64)
x_train, y_train = next(iter(loader_train_0))
x_val, y_val = next(iter(loader_val))

# Data augmentation
augmented_data = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=12,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False)

loader_train = augmented_data.flow(x_train,y_train,batch_size=64)

# Define and train network

model = ResNet18()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)

t0 = time.time()
train_part(model, optimizer, epochs = 10)
print(f'Took {time.time()-t0}s')

# Report test set accuracy
check_accuracy(loader_test, model)

# Save the model
torch.save(model.state_dict(), 'model.pt')

