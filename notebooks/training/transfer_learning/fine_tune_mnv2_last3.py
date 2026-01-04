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
# Settings (fine-tuning uses lower LR)
# ------------------------------------------------------------
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 2
DEVICE = "cpu"

UNFREEZE_LAST_BLOCKS = 3  # fine-tune only the last N blocks of model.features


# ------------------------------------------------------------
# Model builder:
# - Load pretrained MobileNetV2
# - Replace head (4 classes)
# - Freeze backbone then unfreeze last N blocks (partial fine-tune)
# ------------------------------------------------------------
def build_model(num_classes, unfreeze_last_blocks):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # Freeze full backbone first
    for p in model.features.parameters():
        p.requires_grad = False

    # Unfreeze last N blocks (if N > 0)
    if unfreeze_last_blocks > 0:
        last_blocks = list(model.features.children())[-unfreeze_last_blocks:]
        for block in last_blocks:
            for p in block.parameters():
                p.requires_grad = True

    return model


# ------------------------------------------------------------
# Train (one epoch)
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
# Evaluate on validation set (loss + metrics)
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
    print("=== Fine-tuning: MobileNetV2 (unfreeze last blocks) ===")

    # ------------------------------------------------------------
    # Data loading: ImageNet-normalised inputs for pretrained models
    # ------------------------------------------------------------
    dataset, train_subset, val_subset, train_loader, val_loader = get_loaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
        transform_mode="imagenet"
    )

    print(f"Train: {len(train_subset)} | Val: {len(val_subset)}")
    print(f"Batch size: {BATCH_SIZE} | LR: {LR} | Epochs: {EPOCHS}")
    print(f"Unfreeze last blocks: {UNFREEZE_LAST_BLOCKS}")
    print("Classes (label order):", dataset.classes)

    # ------------------------------------------------------------
    # Build model and show how many params will train (sanity check)
    # ------------------------------------------------------------
    model = build_model(num_classes=len(dataset.classes), unfreeze_last_blocks=UNFREEZE_LAST_BLOCKS).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params total: {total_params:,} | trainable: {trainable_params:,}")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    # ------------------------------------------------------------
    # Train + validate loop
    # ------------------------------------------------------------
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
