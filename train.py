"""
Author: Tanay Mehta
LICENSE: Whatever
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import cv2
from tqdm import tqdm

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2

import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

# Load the local scripts
from src.trainer import Trainer
from src.config import Config
from src.dataloader import LDCData
from src.augments import Augments
from src.models import ResNextModel
from src.models import ResNetModel
from src.models import EfficientNetModel
from src.models import VITModel

warnings.simplefilter("ignore")

# Define number of epochs for each fold
nb_epochs=15

if __name__ == "__main__":
    # Get the Device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} for Training the Model...")

    # First, read the data for model training
    data_csv = pd.read_csv("input/train.csv")
    
    # Define Folds for training
    kfold = StratifiedKFold(n_splits=5)

    # Run for KFolds
    for fold_, (trn_idx, val_idx) in enumerate(kfold.split(X=np.zeros(len(data_csv)), y=data_csv['label'])):
        print(f"{'-'*25} Fold: {fold_} {'-'*25}")

        # Define the Model (Change it according to your preference)
        model = ResNextModel(num_classes=5, model_name='resnext50_32x4d').to(device)

        # Define Optimizers and Loss functions
        optim = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=Config.CFG['wd'])
        loss_fn = nn.CrossEntropyLoss().to(device)
        loss_fn_val = nn.CrossEntropyLoss().to(device)

        # Define the Datasets
        train_set = LDCData(df=data_csv.iloc[trn_idx].reset_index(drop=True), augments=Augments.train_augments)
        valid_set = LDCData(df=data_csv.iloc[val_idx].reset_index(drop=True), augments=Augments.valid_augments)
        
        # Define the DataLoaders
        train_loader = DataLoader(
            train_set,
            batch_size=16,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            num_workers=8
        )

        valid_loader = DataLoader(
            valid_set,
            batch_size=32,
            shuffle=False,
            pin_memory=False,
            num_workers=8
        )

        # Define the Trainer
        trainer = Trainer(
            train_dataloader=train_loader,
            valid_dataloader=valid_loader,
            model=model,
            optimizer=optim,
            loss_fn=loss_fn,
            val_loss_fn=loss_fn_val,
            scheduler=None,
            device=device
        )

        # Only when running in interactive mode
        # train_accs = []
        # valid_accs = []
        # train_losses = []
        # valid_losses = []

        # Train for 15 epochs now
        for epoch in range(nb_epochs):
            print(f"{'-'*20} EPOCH: {epoch}/{nb_epochs} {'-'*20}")
            
            # Train for one cycle
            trainer.train_one_cycle()

            # Validate one cycle
            _, _, op_model = trainer.valid_one_cycle()

            # Save MEMORY!
            torch.cuda.empty_cache()

            # Save model with Verbosity
            print(f"Saving Model for this fold...")
            torch.save(op_model.state_dict(), f"resnext50_32x4d_fold_{fold_}_model.pth")

        # Delete temporary variables to save memory in next fold of training
        del train_set, valid_set, train_loader, valid_loader, model, optim, loss_fn, loss_fn_val, trainer
