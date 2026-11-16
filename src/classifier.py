import io
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class ImageClassifier:
    """Pretrained ImageNet classifier using MobileNetV3-Small."""

    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Loading pretrained model on device: {self.device}...")
        self.weights = MobileNet_V3_Small_Weights.DEFAULT
        self.model = mobilenet_v3_small(weights=self.weights).to(self.device)
        self.model.eval()

        # Image preprocessing pipeline matching model training
        self.transform = self.weights.transforms()
        # Categories list (1000 ImageNet classes)
        self.categories = self.weights.meta["categories"]

    def _load_image(self, image_source: str) -> Image.Image:
        """Load image from a local file path or URL."""
        if image_source.startswith("http://") or image_source.startswith("https://"):
            print(f"Fetching remote image from: {image_source}")
            resp = requests.get(
                image_source,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content))
        else:
            image = Image.open(image_source)

        return image.convert("RGB")

    def predict(self, image_source: str, top_k: int = 5):
        """
        Classifies an image from a path or URL.

        Args:
            image_source: Local file path or web URL.
            top_k: Number of highest-confidence predictions to return.

        Returns:
            List of dicts containing rank, label, and confidence score.
        """
        image = self._load_image(image_source)
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output[0], dim=0)

        top_probs, top_indices = torch.topk(probabilities, top_k)

        results = []
        for rank, (idx, prob) in enumerate(zip(top_indices, top_probs), start=1):
            category = self.categories[idx.item()]
            results.append({
                "rank": rank,
                "label": category,
                "confidence": prob.item(),
            })

        return results

