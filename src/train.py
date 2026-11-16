"""
Transfer Learning Boilerplate for Custom Datasets.

Expected dataset structure:
dataset/
    train/
        class_a/
            img1.jpg
        class_b/
            img2.jpg
    val/
        class_a/
        class_b/
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def create_model(num_classes: int):
    # Load pretrained weights
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    # Freeze base feature extractor layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier head for the target number of classes
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model, weights.transforms()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "dataset"

    if not os.path.exists(data_dir):
        print(f"Dataset directory '{data_dir}' not found.")
        print("Please create 'dataset/train' and 'dataset/val' folders with subdirectories per class.")
        return

    num_classes = len(os.listdir(os.path.join(data_dir, "train")))
    model, transform = create_model(num_classes)
    model = model.to(device)

    train_data = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

    print(f"Starting training on {device} for {num_classes} classes...")
    for epoch in range(1, 6):
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch}/5 - Loss: {loss:.4f} - Accuracy: {acc*100:.2f}%")

    os.makedirs("models", exist_ok=True)
    save_path = "models/custom_classifier.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()

