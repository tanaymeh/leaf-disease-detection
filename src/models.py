import torch
import numpy as np
import timm
import torch.nn as nn

class Config:
    CFG = {
        'img_size': 512,
        'tta': 3,
        'wd': 1e-6
    }

class ResNextModel(nn.Module):
    """
    Model Class for ResNext Model Architectures
    """
    def __init__(self, num_classes=5, model_name='resnext50d_32x4d', pretrained=True):
        super(ResNextModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x

class ResNetModel(nn.Module):
    """
    Model Class for ResNet Models
    """
    def __init__(self, num_classes=5, model_name='resnet18', pretrained=True):
        super(ResNetModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x

class EfficientNetModel(nn.Module):
    """
    Model Class for EfficientNet Model
    """
    def __init__(self, num_classes=5, model_name='efficientnet_b5', pretrained=True):
        super(EfficientNetModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class VITModel(nn.Module):
    """
    Model Class for VIT Model
    """
    def __init__(self, num_classes=5, model_name='vit_base_patch16_224', pretrained=True):
        super(VITModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return x