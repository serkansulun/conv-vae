# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#!/bin/python

import argparse
import time
import math as m
import glob
import random
import IPython
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

import model
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

# load images

random_images = False

image_files = glob.glob('data/*.jpg')
if random_images:
    images = random.choices(image_files, k=2)
else:
    images = ['data/000034.jpg', 'data/000081.jpg']
img1 = np.asarray(Image.open(images[0]).resize((112, 128))) / 255.0
img2 = np.asarray(Image.open(images[1]).resize((112, 128))) / 255.0
img1 = np.transpose(img1, [2, 0, 1])
img2 = np.transpose(img2, [2, 0, 1])
img1_v = torch.tensor(img1, dtype=torch.float32).unsqueeze(0).to(device)
img2_v = torch.tensor(img2, dtype=torch.float32).unsqueeze(0).to(device)

# concatanate both images in batch dimension to create network input
img_cat = torch.cat((img1_v, img2_v), 0)

# encode both images
mu, log_var = vae.forward_encoder(img_cat)

# (optional) decode both images
decoded = vae.forward_decoder(mu)
decoded = decoded.to(cpu).numpy()

# print shapes
print('Shapes for image 1:')
print('Raw: ', list(img_cat[0].shape))
print('Encoded: ', list(mu[0].shape))
print('Decoded: ', list(decoded[0].shape))

# interpolate

alpha = 0.5

raw_interp = img1 + alpha * (img2 - img1)
mu_ = (mu[0] + alpha * (mu[1] - mu[0])).unsqueeze(0)
nn_interp = vae.forward_decoder(mu_)
nn_interp = nn_interp.squeeze(0).to(cpu).numpy()

# plot
fig, axes = plt.subplots(2, 3, figsize=np.array([20,15]) * 0.5)

axes[0, 0].imshow(np.transpose(img1, [1, 2, 0]))
axes[0, 0].set_title('Image 1')
axes[0, 0].axis('off')

axes[1, 0].imshow(np.transpose(img2, [1, 2, 0]))
axes[1, 0].set_title('Image 2')
axes[1, 0].axis('off')

axes[0, 1].imshow(np.transpose(decoded[0], [1, 2, 0]))
axes[0, 1].set_title('Encoded-Decoded Image 1')
axes[0, 1].axis('off')

axes[1, 1].imshow(np.transpose(decoded[1], [1, 2, 0]))
axes[1, 1].set_title('Encoded-Decoded Image 2')
axes[1, 1].axis('off')

axes[0, 2].imshow(np.transpose(raw_interp, [1, 2, 0]))
axes[0, 2].set_title('Raw interpolation')
axes[0, 2].axis('off')

axes[1, 2].imshow(np.transpose(nn_interp, [1, 2, 0]))
axes[1, 2].set_title('VAE interpolation')
axes[1, 2].axis('off')
# plt.show()
plt.tight_layout()
plt.savefig('media/out_interp1.png', dpi=300)

# interpolate using different weights

alphas = [0, 0.2, 0.4, 0.6, 0.8, 1]
fig, axes = plt.subplots(2, len(alphas), figsize=np.array([40,15]) * 0.5)

for i, alpha in enumerate(alphas):
    raw_interp = img1 + alpha * (img2 - img1)
    mu_ = (mu[0] + alpha * (mu[1] - mu[0])).unsqueeze(0)
    out_mean = vae.forward_decoder(mu_)
    nn_interp = out_mean.detach().squeeze(0).to(cpu).numpy()

    axes[0, i].imshow(np.transpose(raw_interp, [1, 2, 0]))
    axes[0, i].set_title('Raw - Weight: ' + str(alpha))
    axes[0, i].axis('off')
    axes[1, i].imshow(np.transpose(nn_interp, [1, 2, 0]))
    axes[1, i].set_title('VAE - Weight: ' + str(alpha))
    axes[1, i].axis('off')
plt.tight_layout()
plt.savefig('media/out_interp2.png', dpi=300)

