import os
import pandas as pd
import numpy as np
import torch
from PIL import Image

class Dataset:
    def __init__(self, main_folder, feature, positive=True):
        csv_path = os.path.join(main_folder, 'list_attr_celeba.csv')
        self.photo_folder = os.path.join(main_folder, 'img_align_celeba', 'img_align_celeba')
        if positive:
            label = 1
        else:
            label = -1

        df = pd.read_csv(csv_path)
        self.file_list = list(df[df[feature] == label]['image_id'])
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img = np.asarray(Image.open(os.path.join(self.photo_folder, self.file_list[idx])).resize((112, 128))) / 255.0
        img = np.transpose(img, [2, 0, 1])
        img = torch.tensor(img, dtype=torch.float32)
        return img

