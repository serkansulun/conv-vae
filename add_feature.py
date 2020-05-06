import glob
import numpy as np
import torch
from PIL import Image
import model
import random
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')

remove = False      # removes feature instead of adding it
random_images = False

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

# load feature vector
feature_file = 'eyeglasses.th'
vec = torch.load(feature_file)
vec = vec.to(device)

if remove:
    vec = -0.5 * vec

# load images
images_folder = 'data'
if remove:  # only image with sunglasses
    image = os.path.join(images_folder, '000053.jpg')
elif random_images:
    image_files = glob.glob(images_folder + '/*.jpg')
    image = random.choice(image_files)
else:
    image = os.path.join(images_folder, '000034.jpg')
img_orig = np.asarray(Image.open(image).resize((112, 128))) / 255.0
img = np.transpose(img_orig, [2, 0, 1])
img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

# run
mu, _ = vae.forward_encoder(img)
added = mu + 2* vec     # add feature
decoded = vae.forward_decoder(mu).to(cpu).numpy().squeeze(0).transpose([1, 2, 0])
added_decoded = vae.forward_decoder(added).to(cpu).numpy().squeeze(0).transpose([1, 2, 0])

fig, axes = plt.subplots(1, 3, figsize=np.array([20,9]) * 0.5)

axes[0].imshow(img_orig)
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(decoded)
axes[1].set_title('Decoded')
axes[1].axis('off')

axes[2].imshow(added_decoded)
axes[2].axis('off')

if remove:
    axes[2].set_title('Feature removed')
    outfile = 'out_removed.png'
else:
    axes[2].set_title('Feature added')
    outfile = 'out_added.png'

plt.tight_layout()
plt.savefig('media/' + outfile, dpi=300)