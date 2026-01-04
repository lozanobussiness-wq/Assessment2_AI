import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

# Add notebooks/training to sys.path so we can import pipeline_utils reliably
sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import get_loaders


def build_mobilenetv2(num_classes=4, pretrained=True, freeze_backbone=True):
    # Load MobileNetV2 (pretrained weights are downloaded/cached by PyTorch if needed)
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    # Replace classifier head to match our 4 classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # Freeze backbone for feature extraction
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False

    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    dataset, train_subset, val_subset, train_loader, val_loader = get_loaders(batch_size=32, num_workers=0)

    print("Classes (label order):", dataset.classes)
    print("Class to idx:", dataset.class_to_idx)

    model = build_mobilenetv2(num_classes=len(dataset.classes), pretrained=True, freeze_backbone=True)
    total, trainable = count_parameters(model)
    print(f"Params total: {total:,} | trainable: {trainable:,}")

    images, labels = next(iter(train_loader))
    print("Batch images shape:", images.shape)
    print("Batch labels shape:", labels.shape)

    model.eval()
    with torch.no_grad():
        logits = model(images)
    print("Logits shape:", logits.shape)


if __name__ == "__main__":
    main()
