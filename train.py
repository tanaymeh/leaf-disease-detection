import numpy as np
import pandas as pd

import torch 
import torch.nn as nn

import timm
import torchvision


class ResNextModel(nn.Module):
    """
    Model Class for ResNext Model Architectures
    """
    def __init__(self, num_classes=5, model_name='resnext50d_32x4d'):
        super(ResNextModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x

class ResNetModel(nn.Module):
    """
    Model Class for ResNet Models
    """
    def __init__(self, num_classes=5, model_name='resnet18'):
        super(ResNetModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x

class EfficientNetModel(nn.Module):
    """
    Model Class for EfficientNet Model
    """
    def __init__(self, num_classes=5, model_name='efficientnet_b5'):
        super(ResNetModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x