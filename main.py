import pandas as pd 
import numpy as np
import math 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import pickle 


#prep data
#with open('../data/sampled_filters_train.pkl', 'rb') as f:
#    data = pickle.load(f)
with open('sampled_filters_train.pkl', 'rb') as f:
    data = pickle.load(f)

wavelengths = data['wavelengths']
spectra = data['spectra']
z = data['z']
X = data['X']
df_X = pd.DataFrame(X)
names=data['filter_names']
filter_wls = [int(name[1:-1].rstrip('W')) for name in names]
indices=[0,10,30,500,15000,800000]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#U-net model
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, (diff // 2, (diff + 1) // 2))
        x = torch.cat([x2, x1], dim=1)

        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.convOut = nn.Conv1d(64, 1, kernel_size = 1)

        
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x4 = self.up1(x5, x4)
        x3 = self.up2(x4, x3)
        x2 = self.up3(x3, x2)
        x1 = self.up4(x2, x1)
        x = self.convOut(x1)
        return x

#generate dataset
def add_noise(X, T):
    T += 1
    if T != 1:
        beta1, beta2 = 0.0001, 0.2
        beta_t = np.arange(beta1, beta2 + 0.0001, (beta2 - beta1)/(T - 1) )
    else: 
        beta_t = [0.0001]
    alpha = [1-beta_t[t] for t in range(len(beta_t))]
    alpha_bar = np.prod(alpha[:T-1])

    mu, sigma = math.sqrt(alpha_bar) * X, (1 - alpha_bar) * np.identity(len(X))
    X_tm1 = np.random.multivariate_normal(mu, sigma)
    mu, sigma = math.sqrt(1 - beta_t[T-1]) * X_tm1, (beta_t[T-1]) * np.identity(len(X_tm1))
    X_t = np.random.multivariate_normal(mu, sigma)
    return list(X_t), list(X_tm1)

def gen_dataset(data):
    dataset = {
        "X_t": [],
        "X_tm1": []
    }

    for d in data:
        T = random.randint(1, 100)
        X_t, X_tm1 = add_noise(d, T)
        dataset["X_t"].append(X_t)
        dataset["X_tm1"].append(X_tm1)
    dataset["X_t"] = np.array(dataset["X_t"])
    dataset["X_tm1"] = np.array(dataset["X_tm1"]) 
    return dataset

class MyDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())
        self.length = len(data_dict[self.keys[0]])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        data = {key: self.data_dict[key][index] for key in self.keys}
        return data
    
    
model = UNet().double().to(device)
#loss
loss_fn = nn.MSELoss()
#optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

running_loss_arr = np.array([])
def train_one_epoch(epoch_index, running_loss_arr):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_dataloader):
        # Every data instance is an input + label pair
        X_t, X_tm1 = data.values()
        X_t = X_t.unsqueeze(1).double().to(device)
        X_tm1 = X_tm1.unsqueeze(1).double().to(device)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(X_t)
        
        # Compute the loss and its gradients
        loss = loss_fn(outputs, X_tm1)

        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100 # loss per batch
            tb_x = epoch_index * len(train_dataloader) + i + 1
            running_loss_arr = np.append(running_loss_arr, [tb_x, last_loss])
            running_loss = 0.

    return running_loss_arr


n = 32000
scaler = StandardScaler()
X_normalized = np.array([list(scaler.fit_transform(torch.tensor(X[i]).reshape(-1, 1)).reshape(1, -1)) for i in range(n)]).squeeze()
dataset = MyDataset(gen_dataset(X_normalized[0:n]))

train_dataloader = DataLoader(dataset, batch_size=32, shuffle=False)


epoch_number = 0

EPOCHS = 1


for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    running_loss_arr = train_one_epoch(epoch_number, running_loss_arr)

    # We don't need gradients on to do reporting
    model.train(False)

    
    print('LOSS train: ', running_loss_arr[-1])



    # Track best performance, and save the model's state

    epoch_number += 1


torch.save(model.state_dict(), 'model_param.pt')
df_loss = pd.DataFrame({"time": running_loss_arr.reshape(-1, 2).transpose()[0], "loss": running_loss_arr.reshape(-1, 2).transpose()[1]})
df_loss.to_csv('losses.csv', index = False)


