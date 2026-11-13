import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class ImageClassifier:
    def __init__(self):
        self.device = torch.device("cpu")
