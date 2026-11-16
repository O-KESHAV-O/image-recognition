# Image Recognition Boilerplate

A beginner-friendly, ready-to-run Image Recognition boilerplate using Python, PyTorch, and Torchvision (Pre-trained ResNet-50 / MobileNetV3).

---

## Features

- **Pretrained Inference**: Instant classification using ImageNet weights (1,000 object categories).
- **Top-5 Predictions**: Returns top-5 predicted classes with confidence percentages.
- **Local & URL Support**: Accepts local file paths or web image URLs.
- **Clean Structure**: Modular codebase with preprocessing, inference, and fine-tuning skeleton.

---

## Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/O-KESHAV-O/image-recognition.git
cd image-recognition

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Classify an Image from URL or Local Path

```bash
# Using a web image URL
python src/predict.py --image "https://images.unsplash.com/photo-1543466835-00a7907e9de1"

# Using a local image
python src/predict.py --image sample.jpg
```

Example output:
```text
Loaded model: mobilenet_v3_small (weights=IMAGENET1K_V1)

Top Predictions:
1. golden retriever: 87.42%
2. Labrador retriever: 8.15%
3. cocker spaniel: 1.20%
4. tennis ball: 0.85%
5. kuvasz: 0.31%
```

---

## Project Structure

```
image-recognition/
├── src/
│   ├── __init__.py
│   ├── classifier.py      # Core classification model wrapper
│   └── predict.py         # CLI prediction script
├── requirements.txt       # Project dependencies
├── .gitignore             # Standard Python / ML gitignore
└── README.md              # Project documentation
```

---

## Custom Training

To fine-tune on your own dataset (Transfer Learning), check out `src/train.py`.

