import numpy as np

import torch
import torch.nn as nn

import timm
import torchvision


class LDCModelEffNet(nn.Module):
    """
    Leaf Disease Classification EfficientNet Model Class
    Would work with any version of EfficientNet (B0-B7) as long as using timm library.   
    """
    def __init__(self, num_classes=5, model_arch="tf_efficientnet_b5_ns", pretrained=True):
        """
        Constructor for Model Class.
        
        Args:
            num_classes (int, optional): [Number of Classes in the Dataset]. Defaults to 5.
            model_arch (str, optional): [Correct Name of the Model]. Defaults to "tf_efficientnet_b5_ns".
            pretrained (bool, optional): [True->Training, False->Inference]. Defaults to True.
        """
        super().__init__()
        self.model = timm.create_model(model_name=model_arch, pretrained=pretrained)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
    
    def forward(self, x):
        """Runs the data through the model and returns the predictions

        Args:
            x ([torch.Tensor]): The input transformed and processed image in form of a tensor.
        
        Returns:
            output ([torch.Tensor]): The Output Predictions in form of a tensor.
        """
        output = self.model(x)
        return output