import numpy as np
import pandas as pd

import albumentations
from albumentations.pytorch import ToTensor

from config import Config


class Augments:
    """
    Contains Train, Validation and Testing Augments
    """
    train_augments = albumentations.Compose([
            albumentations.RandomResizedCrop(Config.CFG['img_size'], Config.CFG['img_size']),
            albumentations.Transpose(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.ShiftScaleRotate(p=0.5),
            albumentations.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            albumentations.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            albumentations.CoarseDropout(p=0.5),
            albumentations.Cutout(p=0.5),
            ToTensor()])
    
    valid_augments = albumentations.Compose([
            albumentations.CenterCrop(Config.CFG['img_size'], Config.CFG['img_size'], p=1.),
            albumentations.Resize(Config.CFG['img_size'], Config.CFG['img_size']),
            albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensor()])
    
    test_augments = albumentations.Compose([
            albumentations.RandomResizedCrop(Config.CFG['img_size'], Config.CFG['img_size']),
            albumentations.Transpose(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            albumentations.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            albumentations.ToTensor()])
        
