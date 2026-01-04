import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

# ------------------------------------------------------------
# Path setup: reuse pipeline_utils from notebooks/training
# ------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[1]))
from pipeline_utils import get_loaders


# ------------------------------------------------------------
# Build ResNet18 (pretrained) + new 4-class head
# ------------------------------------------------------------
def build_resnet18(num_classes, freeze_backbone=True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    # Replace final layer (fc) to output our classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Freeze backbone (everything except the new head)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True

    return model


def main():
    # Use ImageNet-normalised transform (same as MobileNet setup)
    dataset, _, _, train_loader, _ = get_loaders(
        batch_size=32,
        num_workers=0,
        transform_mode="imagenet"
    )

    print("Classes (label order):", dataset.classes)
    print("Class to idx:", dataset.class_to_idx)

    model = build_resnet18(num_classes=len(dataset.classes), freeze_backbone=True)

    # Check trainable params are only the head
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params total: {total_params:,} | trainable: {trainable_params:,}")

    # One batch forward pass
    images, labels = next(iter(train_loader))
    print("Batch images shape:", images.shape)
    print("Batch labels shape:", labels.shape)

    model.eval()
    with torch.no_grad():
        logits = model(images)

    print("Logits shape:", logits.shape)


if __name__ == "__main__":
    main()
