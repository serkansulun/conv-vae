import glob
import numpy as np
import torch
from PIL import Image
import pandas as p
import model
import data
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

torch.set_grad_enabled(False)
if torch.cuda.is_available():
    device_name = 'cuda'
else:
    device_name = 'cpu'
device = torch.device(device_name)
cpu = torch.device('cpu')

# load model
dump = torch.load('vae-200.dat', map_location=device_name)
vae = model.VAE(dump['input_shape'], dump['z_dim']).to(device)
vae.load_state_dict(dump['state_dict'])
vae.eval()

z_dim = dump['z_dim']

# data
feature = 'Eyeglasses'
vector = torch.zeros(z_dim, dtype=torch.float32)

for positive in [True, False]:

    dataset = data.Dataset('faces/celeba-dataset', 'Eyeglasses', positive=positive)
    dataset = DataLoader(dataset, batch_size=32, num_workers=4)

    bool_vector = torch.zeros(z_dim, dtype=torch.float32)

    for i, x in enumerate(dataset):
        x = x.to(device)
        mu, _ = vae.forward_encoder(x)
        mu = mu.mean(0).to(cpu)
        vector = (vector * i + mu) / (i+1)
    
    output_name = feature
    if positive:
        vector += bool_vector
    else:
        vector -= bool_vector

    torch.save(vector, os.path.join('vectors', output_name))



