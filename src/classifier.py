import torch
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class ImageClassifier:
    def __init__(self):
        self.device = torch.device("cpu")
        self.weights = MobileNet_V3_Small_Weights.DEFAULT
        self.model = mobilenet_v3_small(weights=self.weights).to(self.device)
        self.model.eval()
        self.transform = self.weights.transforms()
        self.categories = self.weights.meta["categories"]

    def predict_tensor(self, tensor, top_k=5):
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = F.softmax(output[0], dim=0)
        top_probs, top_indices = torch.topk(probabilities, top_k)
        return top_probs, top_indices
