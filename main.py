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
from scipy.linalg import sqrtm

import pickle 
import matplotlib as mpl


##########################################################################################
# Data
##########################################################################################
with open('data/sampled_filters_train.pkl', 'rb') as f:
    data_train = pickle.load(f)
with open('data/sampled_filters_val.pkl', 'rb') as f:
    data_val = pickle.load(f)

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
    
    #get normalized X
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(df_X)
    #get X scaled between [-1, 1]
    min_val = X.min()
    max_val = X.max()
    X_scaled = (X - min_val) / (max_val - min_val)
    y_normalized = scaler.fit_transform(y)

    names=data['filter_names']
    filter_wls = [int(name[1:-1].rstrip('W')) for name in names]
    filter_wls = [element for i, element in enumerate(filter_wls) if i not in outlier_columns]
    return filter_wls, wavelengths, spectra, X, X_normalized, X_scaled, y, y_normalized, z

def rescale_data(X_normalized, X_min, X_max):
    X_rescaled = X_normalized * (X_max - X_min) + X_min
    return X_rescaled

filter_wls, wavelengths, spectra, X, X_normalized, X_scaled,  y, y_normalized, z = get_data(data_train)

_, _, _, vX, vX_normalized, vX_scaled,  vy, vy_normalized, _ = get_data(data_val)


device = "cuda" if torch.cuda.is_available() else "cpu"

#########################################################################################
# Define time_emb
#########################################################################################
class SinPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = time / (10000**(2 * torch.arange(half_dim, device = device) / self.dim))
        embeddings =  torch.cat((embeddings.sin().unsqueeze(1), embeddings.cos().unsqueeze(1)), dim = 1).flatten()
        if self.dim % 2 == 1:
            embeddings =  torch.cat((embeddings, (time / (10000**(2 * torch.tensor([half_dim], device = device) / self.dim))).sin())) 
        
        return embeddings

class SinPosEmbMatrix(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.time_emb = SinPosEmb(dim)

    def forward(self):
        x = torch.tensor([self.time_emb(i).tolist() for i in range(1000)], device = device)
        return x

##########################################################################################
# Define U-Networks
##########################################################################################

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, structure, time_emb_method):
        super(DoubleConv, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.time_emb_method = time_emb_method
        self.structure = structure
        if self.time_emb_method == "features_add":
            for i in range(len(self.structure)):
                if out_channels == self.structure[i]:
                    time_emb = SinPosEmbMatrix(31 // (2**i))
                    self.time_emb = time_emb()
        if self.time_emb_method == "channels_add":
            time_emb = SinPosEmbMatrix(256)
            self.time_emb = time_emb()
            self.linear = nn.Linear(256, out_channels)


        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.in_channels = in_channels
        
    def forward(self, x, t):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)


        if self.time_emb_method == "features_add":
            time = self.time_emb[t, :].unsqueeze(1)
            x = x + time
        elif self.time_emb_method == "channels_add":
            time = self.time_emb[t, :].unsqueeze(1)
            time = self.linear(time).squeeze(1).unsqueeze(-1)
            x = x + time

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, structure, time_emb_method):
        super(Down, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels, structure, time_emb_method)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, x, t):
        x = self.pool(x)
        x = self.conv(x, t)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, structure, time_emb_method):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, structure, time_emb_method)
        
    def forward(self, x1, x2, t):
        x1 = self.up(x1)
        diff = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, (diff // 2, (diff + 1) // 2))
        x = torch.cat([x2, x1], dim=1)

        x = self.conv(x, t)
        return x

        
class BigUNet(nn.Module):
    def __init__(self, structure, time_emb_method = "features_add"):
        super(BigUNet, self).__init__()
        self.structure = structure
        self.time_emb_method = time_emb_method
        assert self.time_emb_method in ["features_add", "channels_add"]

        self.convIn = DoubleConv(1, self.structure[0], self.structure, self.time_emb_method)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        for i in range(len(self.structure) - 1):
            self.downs.append(Down(self.structure[i], self.structure[i+1], self.structure, self.time_emb_method))
        for i in range(len(self.structure) - 1):
            self.ups.append(Up(self.structure[i+1], self.structure[i], self.structure, self.time_emb_method))

        self.convOut = nn.Conv1d(self.structure[0], 1, kernel_size = 1)


    def forward(self, x, t):
        skips = []
        x = self.convIn(x, t)

        for i in range(len(self.structure)-1):
            skips.append(x)
            x = self.downs[i](x, t)
        
        
        for i in reversed(range(len(self.structure) - 1)):
            x = self.ups[i](x, skips[i], t)
        
        x = self.convOut(x)
        
        return x


##########################################################################################
# Forward diffusion process and Dataset generating
##########################################################################################

def cos_betaAlpha_schedule(T = 1000, s=0.005):
    x = torch.linspace(0, T, T+1)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0.0001, 0.9999)
    alpha_bar = torch.cumprod(torch.tensor([1-betas[i] for i in range(len(betas))]), dim = 0)
    return betas, alpha_bar

def extract(a, t, x_shape):
    b, *_ = t.shape
    
    a = a.to(t.device)
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def add_noise(X, t, alpha_bar):
    eps = torch.randn_like(X)
    mean = torch.sqrt(extract(alpha_bar.to(t.device), t, X.shape)) * X
    std = torch.sqrt(1 - extract(alpha_bar.to(t.device), t, X.shape))
    sample = mean + std * eps
    return sample, eps

""" def add_noise(X, t, alpha_bar):
    eps = torch.randn_like(X)

    mean = torch.sqrt(alpha_bar[t][:, None]) * X
    std = torch.sqrt(1 - alpha_bar[t][:, None])
    sample = mean + std * eps
    return sample, eps
 """

def denoise(model, n, dim = 31, T = 1000):
    X = torch.randn(n, dim, device = device)
    betas, alpha_bar = cos_betaAlpha_schedule()
    betas = betas.to(device)
    alpha_bar = alpha_bar.to(device)
    for tt in range(T - 1, -1, -1):
        t = torch.tensor([tt] * n, device = device)
        
        z = torch.randn(n, dim, device = device) if tt > 1 else 0
        X_noise = model(X.unsqueeze(1), t).detach().squeeze()

        t = t.long()

        X = (1 / torch.sqrt(1 - betas[tt])) * (X - (betas[tt]/torch.sqrt(1 - alpha_bar[tt]) * X_noise)) + torch.sqrt(betas[tt]) * z
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

class TestModel():
    def __init__(self) -> None:
        pass

    def get_one_data(self):
        foo = torch.tensor(X_scaled[0]).unsqueeze(0).unsqueeze(0).to(device)
        t = torch.randint(0, 1000, (1,), device=device).long()
        return foo, t


#########################################################################################
# Validation functions
#########################################################################################
class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(192, 192)
        self.fc2 = nn.Linear(192, 4)  
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x2 = self.fc1(x)
        x = self.relu(x2)
        x = self.fc2(x)

        return x2, x
    
class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(384, 256)
        self.fc2 = nn.Linear(256, 4)  
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x2 = self.flatten(x)
        x = self.fc1(x2)
        x = self.relu(x)
        x = self.fc2(x)

        return x2, x

def poly_mmd2(f_of_X, f_of_Y, d=3, alpha=1.0, c=1.0):
    K_XX = (alpha * (f_of_X @ f_of_X.t()) + c)
    K_XX_mean = torch.mean(K_XX.pow(d))

    K_YY = (alpha * (f_of_Y @ f_of_Y.t()) + c)
    K_YY_mean = torch.mean(K_YY.pow(d))

    K_XY = (alpha * (f_of_X @ f_of_Y.t()) + c)
    K_XY_mean = torch.mean(K_XY.pow(d))

    return K_XX_mean + K_YY_mean - 2 * K_XY_mean 

from sklearn.metrics.pairwise import rbf_kernel

def rbf_mmd2(f_of_X, f_of_Y, gamma = 0.23):
    f_of_X = f_of_X.detach().cpu()
    f_of_Y = f_of_Y.detach().cpu()
    K_XX_mean = np.mean(rbf_kernel(f_of_X, f_of_X, gamma = gamma))
    K_XY_mean = np.mean(rbf_kernel(f_of_X, f_of_Y, gamma = gamma))
    K_YY_mean = np.mean(rbf_kernel(f_of_Y, f_of_Y, gamma = gamma))
    return K_XX_mean + K_YY_mean - 2 * K_XY_mean

def calculate_fid(real_embeddings, generated_embeddings):
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings,  rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

 
def validate_model(model, inc_model, bs = 4096):
    training = model.training
    model.train(False)
    tensor_model = denoise(model, bs).unsqueeze(1)
    model.train(training)
    tensor_source = torch.tensor(vX_scaled[np.random.randint(0, len(vX_scaled), bs)]).unsqueeze(1).to(device)
    inc_model_out, out1 = inc_model(tensor_model)
    inc_source_out, out2 = inc_model(tensor_source)

    return calculate_fid(inc_source_out.detach().cpu().numpy(), inc_model_out.detach().cpu().numpy()), rbf_mmd2(inc_model_out.squeeze(), inc_source_out.squeeze()).item()

##########################################################################################
# Train model(s)
##########################################################################################
def train_model(model, inc_model, n = len(X_normalized), T = 1000, batch_size = 256, EPOCHS = 35):
    #loss
    loss_fn = nn.MSELoss()
    #optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    beta_t, alpha_bar = cos_betaAlpha_schedule(T = T)
    #loss data
    running_loss_arr = []
    running_fid = [[0] for i in range((EPOCHS)+ 1)]
    fid, mmd = validate_model(model, inc_model)
    running_fid[0] = [0, fid, mmd]
    
    #training
    model.train(True)
    with open(path + "foo.txt", "w") as f:
        f.write("start \n")
    for epoch in range(EPOCHS):
        with open(path + "foo.txt", "a") as f:
            f.write("epoch: " + str(epoch) + "\n")
        # generate data_loader
        train_dataloader = DataLoader(MyDataset({"X": torch.tensor(X_scaled[:n])}), batch_size=batch_size, shuffle=True, drop_last= False)
        
        
        print('EPOCH ', epoch, ":")
        
        running_loss = 0
        last_loss = 0

        for i, data in enumerate(train_dataloader):
            b, _= data['X'].shape
            #t = np.random.randint(1, T + 1)
            t = torch.randint(0, 1000, (b,), device=device).long()

            X_t, X_noise = add_noise(data['X'].to(device), t, alpha_bar)
            X_t = X_t.unsqueeze(1).to(device)
            X_noise = X_noise.unsqueeze(1).to(device)
            # Zero gradients for every batch
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(X_t, t)
            
            # Compute the loss and its gradients
            loss = loss_fn(outputs, X_noise)
            
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data
            running_loss += loss.item()
            if (i+1) % 200 == 0:
                last_loss = running_loss / 200 # loss per batch
                tb_x = epoch * len(train_dataloader) + i + 1
                running_loss_arr.append([tb_x, last_loss])
                running_loss = 0.
                with open(path + "foo.txt", "a") as f:
                    f.write(str(last_loss) + "\n")
        fid, mmd = validate_model(model, inc_model)
        running_fid[epoch + 1] = [epoch + 1, fid, mmd]
        scheduler.step()

        print('LOSS train: ', running_loss_arr[-1])

        #get FID data 
        #fid, mmd = validate_model(model, inc_model)
        #running_fid[epoch + 1] = [epoch, fid, mmd]

    model.train(False)
    return np.array(running_loss_arr), np.array(running_fid)


struct = [64, 128, 256, 512]
model_big_featPos = BigUNet(structure = struct, time_emb_method = "features_add").to(device)
model_big_chanPos = BigUNet(structure = struct, time_emb_method = "channels_add").to(device)
struct = [64, 128, 256]
model_med_featPos = BigUNet(structure = struct, time_emb_method = "features_add").to(device)
model_med_chanPos = BigUNet(structure = struct, time_emb_method = "channels_add").to(device)
struct = [32, 64, 128]
model_small_featPos = BigUNet(structure = struct, time_emb_method = "features_add").to(device)
model_small_chanPos = BigUNet(structure = struct, time_emb_method = "channels_add").to(device)

modelList_featPos = [model_big_featPos, model_med_featPos, model_small_featPos]
modelNames_featPos = ["model_big_featPos", "model_med_featPos", "model_small_featPos"]
modelList_chanPos = [model_big_chanPos, model_med_chanPos, model_small_chanPos]
modelNames_chanPos = ["model_big_chanPos", "model_med_chanPos", "model_small_chanPos"]

modelList = modelList_featPos + modelList_chanPos
modelNames = modelNames_featPos + modelNames_chanPos

model_CNN1 = CNN2().to(device)
model_CNN1.load_state_dict(torch.load('runsCNN2/model_param_model_CNN.pt', map_location= device))
model_CNN1.train(False)

model_CNN2 = CNN2().to(device)
model_CNN2.load_state_dict(torch.load('runsCNN2/model_param_model_CNN.pt', map_location= device))
model_CNN2.train(False)


##########################################################################################
# train all models
##########################################################################################
T = 1000
n = len(X_scaled)
batch_size = 512
epochs = 50
path = "runs8/"


for name, model in zip(modelNames, modelList):
    print("#####################################")
    print("n: ", n)
    print("batch_size: ", batch_size)
    print("Epochs: ", epochs)
    #print("model: ", model)
    
    running_loss_arr, running_fid = train_model(model, model_CNN2, n = n, T = T, batch_size = batch_size, EPOCHS = epochs)

    torch.save(model.state_dict(), path + "model_param_" + name + ".pt")
    df_loss = pd.DataFrame({"time": running_loss_arr.reshape(-1, 2).transpose()[0], "loss": running_loss_arr.reshape(-1, 2).transpose()[1]})
    df_loss.to_csv(path + "losses" + name + ".csv", index = False)
    df_fid = pd.DataFrame(running_fid, columns=['epoch', 'fid', 'mmd'])
    df_fid.to_csv(path + "fid" + name + ".csv", index = False)
    
    print("#####################################")
