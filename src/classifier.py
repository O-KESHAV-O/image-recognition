import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class ImageClassifier:
    def __init__(self):
        self.device = torch.device("cpu")
        self.weights = MobileNet_V3_Small_Weights.DEFAULT
        self.model = mobilenet_v3_small(weights=self.weights).to(self.device)
        self.model.eval()
        self.transform = self.weights.transforms()
        self.categories = self.weights.meta["categories"]
