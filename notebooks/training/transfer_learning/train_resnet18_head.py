import sys
from pathlib import Path
from time import time

import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ------------------------------------------------------------
# Path setup: reuse pipeline_utils from notebooks/training
# ------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[1]))
from pipeline_utils import get_loaders


# ------------------------------------------------------------
# Run settings
# ------------------------------------------------------------
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 3
DEVICE = "cpu"


# ------------------------------------------------------------
# Build ResNet18 pretrained + new 4-class head (freeze backbone)
# ------------------------------------------------------------
def build_resnet18(num_classes):
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Freeze everything except final layer
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    return model


# ------------------------------------------------------------
# Train for one epoch
# ------------------------------------------------------------
def train_one_epoch(model, loader, optimizer):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


# ------------------------------------------------------------
# Evaluate on validation set
# ------------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, acc, macro_f1, cm


def main():
    print("=== Transfer Learning: ResNet18 (train head + validate) ===")

    # Use ImageNet normalisation for pretrained models
    dataset, train_subset, val_subset, train_loader, val_loader = get_loaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
        transform_mode="imagenet"
    )

    print(f"Train: {len(train_subset)} | Val: {len(val_subset)}")
    print(f"Batch size: {BATCH_SIZE} | LR: {LR} | Epochs: {EPOCHS}")
    print("Classes (label order):", dataset.classes)

    model = build_resnet18(num_classes=len(dataset.classes)).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params total: {total_params:,} | trainable: {trainable_params:,}")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    t0 = time()

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss, acc, macro_f1, cm = evaluate(model, val_loader)

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"Train avg loss: {train_loss:.4f}")
        print(f"Val avg loss:   {val_loss:.4f}")
        print(f"Accuracy:       {acc:.4f}")
        print(f"Macro F1:       {macro_f1:.4f}")
        print("\nConfusion matrix (rows=true, cols=pred):")
        print(cm)

    print(f"\nElapsed (s): {time() - t0:.1f}")


if __name__ == "__main__":
    main()
