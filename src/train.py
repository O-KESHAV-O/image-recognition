import os
import torch
import torch.nn as nn
from torchvision import models

def create_model(num_classes: int):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    return model, weights.transforms()
