import os

import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn


class LeafDataset(Dataset):
    """
    Dataset class for Cassava Leaf Disease Data
    """
    def __init__(self, data, transforms=None, data_folder="../input/cassava-leaf-disease-classification/train_images", op_labels=True, do_fmix=True, do_cutmix=False):
        super().__init__()
        self.data = data.sample(frac=1).reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_folder = data_folder
        self.op_labels = op_labels
        self.do_fmix = do_fmix
        self.do_cutmix = do_cutmix

        # Return labels only if that's the mode (train/val/test)
        if self.op_labels == True:
            self.labels = self.data['label'].values

        self.data['image_id'] = self.data['image_id'].apply(lambda x: os.path.join(self.data_folder, x))

        # Sanity check that data path exists
        try:
            os.path.exists(self.data['image_id'][0])
        except Exception as e:
            print(f"There's some problem with the paths.\nThis does not exist: {self.data['image_id'][0]}")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        Returns the (img, target) pair at the given index (idx).
        TODO: Implement the Mix.cutmix() and Mix.fmix() function calls for the dataloader
        
        Args:
            idx ([int]): Index to access an element of the target list

        Returns:
            img ([np.ndarray]): Returns a transformed image in form of Numpy array
            target ([np.ndarray]) (optional): Returns a numpy array of integer labels
        """
        if self.op_labels:
            target = self.labels[idx]

        # Read the image and convert it from BGR to RGB
        img = cv2.imread(self.data['image_id'][idx])
        img = img[:, :, ::-1]
        
        if self.transforms:
            img = self.transforms(image=img)['image']

        if self.op_labels == True:
            return img, target
        else:
            return img