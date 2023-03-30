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

import pickle 

with open('data/sampled_filters_train.pkl', 'rb') as f:
    data = pickle.load(f)

def get_data(data):
    #in data
    #wavelength (4000000, 149)
    #spectra (4000000, 149)
    #X (4000000, 37)
    #y (4000000, 4)
    #z (4000000, 1)
    #zmin (float)
    #zmax (float)
    #filter_names (string list), len = 37
    df_X = pd.DataFrame(data['X'])
    outlier_columns = df_X.columns[df_X.gt(58.8).any(axis = 0)]
    df_X = df_X.drop(outlier_columns, axis = 1)
    X = df_X.to_numpy()
    y = pd.DataFrame(data['y'])
    z = pd.DataFrame(data['z'])
    wavelengths = data['wavelengths']
    spectra = data['spectra']
    
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(df_X)

    names=data['filter_names']
    filter_wls = [int(name[1:-1].rstrip('W')) for name in names]
    filter_wls = [element for i, element in enumerate(filter_wls) if i not in outlier_columns]
    indices=[0,10,30,500,15000,800000]
    return filter_wls, wavelengths, spectra, X, X_normalized, y, z
filter_wls, wavelengths, spectra, X, X_normalized, y, z = get_data(data)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

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

class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SingleConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
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

class BigUNet(nn.Module):
    def __init__(self):
        super(BigUNet, self).__init__()
        self.conv = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.convOut = nn.Conv1d(64, 1, kernel_size = 1)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x3 = self.up2(x4, x3)
        x2 = self.up3(x3, x2)
        x1 = self.up4(x2, x1)
        x = self.convOut(x1)
        return x
    
class MedBigUNet(nn.Module):
    def __init__(self):
        super(MedBigUNet, self).__init__()
        self.conv = DoubleConv(1, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        self.convOut = nn.Conv1d(32, 1, kernel_size = 1)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x3 = self.up2(x4, x3)
        x2 = self.up3(x3, x2)
        x1 = self.up4(x2, x1)
        x = self.convOut(x1)
        return x

class MediumUNet(nn.Module):
    def __init__(self):
        super(MediumUNet, self).__init__()
        self.conv = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.convOut = nn.Conv1d(64, 1, kernel_size = 1)
        
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x2 = self.up1(x3, x2)
        x1 = self.up2(x2, x1)
        x = self.convOut(x1)
        return x

class SmallUNet(nn.Module):
    def __init__(self):
        super(SmallUNet, self).__init__()
        self.conv = DoubleConv(1, 32)
        self.down1 = DoubleConv(32, 64)
        self.down2 = DoubleConv(64, 128)
        self.up1 = Up(128, 64)
        self.up2 = Up(64, 32)
        self.convOut = nn.Conv1d(32, 1, kernel_size = 1)
        
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x2 = self.up1(x3, x2)
        x1 = self.up2(x2, x1)
        x = self.convOut(x1)
        return x


def add_noise(X, t, beta_t, alpha, alpha_bar, T = 100):

    mu, sigma = math.sqrt(alpha_bar[t]) * X, (1 - alpha_bar[t]) * np.identity(len(X))
    X_tm1 = np.random.multivariate_normal(mu, sigma)
    mu, sigma = math.sqrt(1 - beta_t[t]) * X_tm1, (beta_t[t]) * np.identity(len(X_tm1))
    X_t = np.random.multivariate_normal(mu, sigma)
    pred_noise =  X_t - X_tm1
    return list(X_t), list(pred_noise)

def gen_dataset(data):
    T = 100

    X_t_arr = []
    X_tm1_arr = []

    beta_t = np.array([0.0001 * (1.05**i) for i in range(T)])
    alpha = np.array([1-beta_t[i] for i in range(len(beta_t))])
    alpha_bar = np.cumprod(alpha)

    for d in data:
        t = random.randint(0, 99)
        X_t, X_tm1 = add_noise(d, t, beta_t, alpha, alpha_bar)
        X_t_arr.append(X_t)
        X_tm1_arr.append(X_tm1)

    return X_t_arr, X_tm1_arr

def denoise(model, X, T = 100):
    X = torch.tensor(X).unsqueeze(0).unsqueeze(0).to(device)
    for _ in range(T):
        X_noise = model(X) 
        X = X - X_noise
    return X

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
    
def print_metaData(n, batch_size):
    print(n, batch_size)


def train_model(model, n = len(X_normalized), batch_size = 256, EPOCHS = 50):
    #loss
    loss_fn = nn.MSELoss()
    #optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    #loss data
    running_loss_arr = []

    #training
    model.train(True)
    for epoch in range(EPOCHS):
        # generate data_loader
        dataset = MyDataset(gen_dataset(X_normalized[0:n]))
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        
        
        print('EPOCH ', epoch, ":")
        
        running_loss = 0
        last_loss = 0

        for i, data in enumerate(train_dataloader):
            # Every data instance is an input + label pair
            X_t, X_tm1 = data.values()
            X_t = X_t.unsqueeze(1).double().to(device)
            X_tm1 = X_tm1.unsqueeze(1).double().to(device)
            # Zero gradients for every batch
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(X_t)
            
            # Compute the loss and its gradients
            loss = loss_fn(outputs, X_tm1)

            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data
            running_loss += loss.item()
            if (i+1) % 100 == 0:
                last_loss = running_loss / 100 # loss per batch
                tb_x = epoch * len(train_dataloader) + i + 1
                running_loss_arr.append([tb_x, last_loss])
                running_loss = 0.

        scheduler.step()

        print('LOSS train: ', running_loss_arr[-1])

    model.train(False)
    return np.array(running_loss_arr)

def run():
    n = 64000
    batch_size = 64
    epochs = 1

    model_big = BigUNet().double().to(device)
    model_medBig = MedBigUNet().double().to(device)
    model_medium = MediumUNet().double().to(device)
    model_small = SmallUNet().double().to(device)
    modelList = [model_big, model_medBig, model_medium, model_small]
    modelNames = ["model_big", "model_medBig", "model_medium", "model_small"]

    for name, model in zip(modelNames, modelList):
        print("#####################################")
        print("n: ", n)
        print("batch_size: ", batch_size)
        print("Epochs: ", epochs)
        print("model: ", model)

        running_loss_arr = train_model(model, n = n, batch_size = batch_size, EPOCHS = epochs)

        torch.save(model.state_dict(), 'output/model_param_' + name + '.pt')
        df_loss = pd.DataFrame({"time": running_loss_arr.reshape(-1, 2).transpose()[0], "loss": running_loss_arr.reshape(-1, 2).transpose()[1]})
        df_loss.to_csv('output/losses' + name + '.csv', index = False)
        print("#####################################")


if __name__ == "__main__":
    run()