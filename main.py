
##########################################################################################
# Import libraries
##########################################################################################
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


##########################################################################################
# Data
##########################################################################################
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
    return filter_wls, wavelengths, spectra, X, X_normalized, y, z

filter_wls, wavelengths, spectra, X, X_normalized, y, z = get_data(data)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

##########################################################################################
# Define U-Networks
##########################################################################################
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
    

##########################################################################################
# Forward diffusion process and Dataset generating
##########################################################################################
def cos_betaAlpha_schedule(T = 300, s=0.005):
    x = torch.linspace(0, T, T+1)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0.0001, 0.9999)
    alpha_bar = np.cumprod(np.array([1-betas[i] for i in range(len(betas))]))
    return betas, alpha_bar


def add_noise(X, t, alpha_bar):
    eps = torch.randn_like(X)
    mean = math.sqrt(alpha_bar[t - 1]) * X
    std = math.sqrt(1 - alpha_bar[t - 1])
    sample = mean + std * eps
    return sample, eps


def denoise(model, X, T = 300):
    betas, alpha_bar = cos_betaAlpha_schedule()

    X = torch.tensor(X)
    X_noise = torch.zeros(len(X))
    for t in range(T-1, -1, -1):
        mu_t = (1 / math.sqrt(1 - betas[t])) * (X - (betas[t])/(math.sqrt(1 - alpha_bar[t])) * X_noise)
        X = torch.tensor(np.random.multivariate_normal(mu_t, betas[T - 1] * np.identity(31)))

        X_noise = model(X.unsqueeze(0).unsqueeze(0).to(device))
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
##########################################################################################
# Train model(s)
##########################################################################################
def train_model(model, n = len(X_normalized), T = 300, batch_size = 256, EPOCHS = 50):
    #loss
    loss_fn = nn.MSELoss()
    #optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    beta_t, alpha_bar = cos_betaAlpha_schedule(T = T)
    #loss data
    running_loss_arr = []

    #training
    model.train(True)
    with open("../runs/foo.txt", "w") as f:
        f.write("start \n")
    for epoch in range(EPOCHS):
        with open("../runs/foo.txt", "a") as f:
            f.write("epoch: " + str(epoch) + "\n")
        # generate data_loader
        train_dataloader = DataLoader(MyDataset({"X": torch.tensor(X_normalized[:n])}), batch_size=batch_size, shuffle=False)
        
        
        print('EPOCH ', epoch, ":")
        
        running_loss = 0
        last_loss = 0

        for i, data in enumerate(train_dataloader):
            # Every data instance is an input + label pair
            t = np.random.randint(1, T + 1)
            X_t, X_noise = add_noise(data['X'], t, alpha_bar)
            X_t = X_t.unsqueeze(1).double().to(device)
            X_noise = X_noise.unsqueeze(1).double().to(device)
            # Zero gradients for every batch
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(X_t)
            
            # Compute the loss and its gradients
            loss = loss_fn(outputs, X_noise)
            
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
                with open("../runs/foo.txt", "a") as f:
                    f.write(str(last_loss) + "\n")
        scheduler.step()

        print('LOSS train: ', running_loss_arr[-1])

    model.train(False)
    return np.array(running_loss_arr)


##########################################################################################
# run
##########################################################################################
def run():
    n = len(X_normalized)
    batch_size = 512
    epochs = 30

    model_big = BigUNet().double().to(device)
    model_medBig = MedBigUNet().double().to(device)
    model_medium = MediumUNet().double().to(device)
    model_small = SmallUNet().double().to(device)
    modelList = [model_big, model_medBig, model_medium, model_small]
    modelNames = ["model_big", "model_medBig", "model_medium", "model_small"]
    modelList = [model_big]
    modelNames = ["model_big"]

    for name, model in zip(modelNames, modelList):
        print("#####################################")
        print("n: ", n)
        print("batch_size: ", batch_size)
        print("Epochs: ", epochs)
        print("model: ", model)

        running_loss_arr = train_model(model, n = n, batch_size = batch_size, EPOCHS = epochs)

        torch.save(model.state_dict(), 'runs/model_param_' + name + '.pt')
        df_loss = pd.DataFrame({"time": running_loss_arr.reshape(-1, 2).transpose()[0], "loss": running_loss_arr.reshape(-1, 2).transpose()[1]})
        df_loss.to_csv('runs/losses' + name + '.csv', index = False)
        print("#####################################")


if __name__ == "__main__":
    run()
















































