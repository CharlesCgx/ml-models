#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Pytorch Variational Auto-Encoder (VAE) ###

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Variational Auto-Encoder

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(28*28, 400)
        self.relu = nn.ReLU()
        self.fcMu = nn.Linear(400, latent_dim)
        self.fcLogvar = nn.Linear(400, latent_dim)
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 28*28)
        self.sigmoid = nn.Sigmoid()
        
    def encode(self, x):
        tmp = self.relu(self.fc1(x))
        return self.fcMu(tmp), self.fcLogvar(tmp)
    
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)
        
    def decode(self, z):
        return self.sigmoid(self.fc4(self.relu(self.fc3(z))))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 28*28))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar, z

def loss_function_VAE(x_recon, x, mu, logvar):
        bce = F.binary_cross_entropy(x_recon, x.view(-1, 784))
        kld = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1).mean()
        kld /= batch_size * 28 * 28
        return bce, kld

### Example with MNIST dataset ###

def show(img):
    if torch.cuda.is_available():
        img = img.cpu()
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))

# Device selection
GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

# Data loading
if not os.path.exists('./CW_VAE/MNIST'):
    os.makedirs('./CW_VAE/MNIST')
    
train_dat = datasets.MNIST("data/", train=True, download=True, transform=transform)
test_dat = datasets.MNIST("data/", train=False, transform=transform)
loader_train = DataLoader(train_dat, batch_size, shuffle=True)
loader_test = DataLoader(test_dat, batch_size, shuffle=False)
sample_inputs, _ = next(iter(loader_test))
fixed_input = sample_inputs[:32, :, :, :]

save_image(fixed_input, './CW_VAE/MNIST/image_original.png')

# Hyperparameters
num_epochs = 10
learning_rate  =  1e-3
batch_size = 128
latent_dim = 20

transform = transforms.Compose([transforms.ToTensor(),])
denorm = lambda x:x

# Model definition
model = VAE().to(device)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters is: {}".format(params))
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
train_BCE, train_KLD = [], []
test_BCE, test_KLD = [], []


for epoch in range(num_epochs):     
    model.train()
    train_loss_bce = 0
    train_loss_kld = 0
    for batch_idx, (x_true, _) in enumerate(loader_train):
      optimizer.zero_grad()
      x_true = x_true.to(device)
      x_recon, mu, logvar, _ = model(x_true)
      bce, kld = loss_function_VAE(x_recon, x_true, mu, logvar)
      loss = bce + 10*kld
      loss.backward()
      train_loss_bce += bce.item()
      train_loss_kld += 10*kld.item()
      optimizer.step()
    
    print('Epoch: {} Average loss: {:.4f}'.format(
      epoch, (train_loss_kld + train_loss_bce) / len(loader_train.dataset)))
    train_BCE.append(train_loss_bce/ len(loader_train.dataset))
    train_KLD.append(train_loss_kld/ len(loader_train.dataset))

    model.eval()
    test_loss_bce = 0
    test_loss_kld = 0
    with torch.no_grad():
      for batch_idx, (x_true, _) in enumerate(loader_test):
        x_true = x_true.to(device)
        x_recon, mu, logvar, _ = model(x_true)
        bce, kld = loss_function_VAE(x_recon, x_true, mu, logvar)
        test_loss_bce += bce.item()
        test_loss_kld += 10*kld.item()

      print('> Test set loss: {:.4f}'.format((test_loss_kld + test_loss_bce)/len(loader_test.dataset)))
    test_BCE.append(test_loss_bce/len(loader_test.dataset))
    test_KLD.append(test_loss_kld/len(loader_test.dataset))

# Save the model 
torch.save(model.state_dict(), './CW_VAE/MNIST/VAE_model.pth')

