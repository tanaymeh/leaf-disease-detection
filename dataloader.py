import os

import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from config import Config


class LDCData(Dataset):
    def __init__(self, df, num_classes=5, is_train=True, augments=None, img_size=Config.CFG['img_size'], img_path="../input/cassava-leaf-disease-classification/train_images/"):
        super().__init__()
        self.df = df.sample(frac=1).reset_index(drop=True)
        self.num_classes = num_classes
        self.is_train = is_train
        self.augments = augments
        self.img_size = img_size
        self.img_path = img_path
        
        # Add the Right Image Path
        self.df['image_id'] = self.df['image_id'].apply(lambda x: os.path.join(self.img_path, x))
    
    def __getitem__(self, idx):
        # Read the image, Resize, convert to RGB from BGR
        img = cv2.imread(self.df['image_id'][idx])
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img[:, :, ::-1]
        
        # Augments must be albumentations
        if self.augments:
            img = self.augments(image=img)['image']
        
        if self.is_train:
            label = self.df['label'][idx]
            return img, label
        
        return img
    
    def __len__(self):
        return len(self.df)