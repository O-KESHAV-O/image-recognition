import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class ImageClassifier:
    def __init__(self):
        self.device = torch.device("cpu")
        self.weights = MobileNet_V3_Small_Weights.DEFAULT
        self.model = mobilenet_v3_small(weights=self.weights).to(self.device)
        self.model.eval()
        self.transform = self.weights.transforms()
        self.categories = self.weights.meta["categories"]

    def _load_image(self, image_path: str) -> Image.Image:
        image = Image.open(image_path)
        return image.convert("RGB")

    def predict(self, image_path: str, top_k: int = 5):
        image = self._load_image(image_path)
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output[0], dim=0)
        top_probs, top_indices = torch.topk(probabilities, top_k)
        results = []
        for rank, (idx, prob) in enumerate(zip(top_indices, top_probs), start=1):
            results.append({"rank": rank, "label": self.categories[idx.item()], "confidence": prob.item()})
        return results
