import sys
from pathlib import Path
from time import time

import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ------------------------------------------------------------
# Path setup:
# We run scripts from different folders, so we add notebooks/training
# to sys.path to reuse pipeline_utils without packaging the project.
# ------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import get_loaders


# ------------------------------------------------------------
# Run settings (simple and explicit)
# ------------------------------------------------------------
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 3          # Start with 1 epoch to validate end-to-end correctness
DEVICE = "cpu"


# ------------------------------------------------------------
# Model builder:
# - Load pretrained MobileNetV2 (ImageNet)
# - Replace final classifier to output 4 classes
# - Freeze backbone so we train only the new head (feature extraction)
# ------------------------------------------------------------
def build_model(num_classes, freeze_backbone=True):
    # Load MobileNetV2 pretrained weights (download once, cached afterwards)
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)

    # Replace the final linear layer: 1000 classes (ImageNet) -> our 4 classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # Freeze feature extractor so only the classifier head learns
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False

    return model


# ------------------------------------------------------------
# Training for one epoch:
# - Forward pass
# - Compute CrossEntropy loss
# - Backpropagate and update ONLY trainable parameters (the head)
# ------------------------------------------------------------
def train_one_epoch(model, loader, optimizer):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0

    for images, labels in loader:
        # Move data to device (CPU in this project)
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Standard training step
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        # Track loss weighted by batch size
        total_loss += loss.item() * images.size(0)

    # Average loss over all samples in the training subset
    return total_loss / len(loader.dataset)


# ------------------------------------------------------------
# Evaluation on validation set:
# - No gradients (faster, less memory)
# - Collect predictions and labels
# - Compute: average loss, accuracy, macro F1, confusion matrix
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

    # Average loss over all samples in the validation subset
    avg_loss = total_loss / len(loader.dataset)

    # Core metrics (consistent with baseline comparison)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, acc, macro_f1, cm


def main():
    print("=== Transfer Learning: MobileNetV2 (train head + validate) ===")

    # ------------------------------------------------------------
    # Data loading:
    # Reuse the same split logic as baseline.
    # IMPORTANT: use ImageNet-normalised inputs for pretrained models.
    # ------------------------------------------------------------
    dataset, train_subset, val_subset, train_loader, val_loader = get_loaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
        transform_mode="imagenet"
    )

    # Quick run context (useful for logbook output)
    print(f"Train: {len(train_subset)} | Val: {len(val_subset)}")
    print(f"Batch size: {BATCH_SIZE} | LR: {LR} | Epochs: {EPOCHS}")
    print("Classes (label order):", dataset.classes)

    # ------------------------------------------------------------
    # Model + optimizer:
    # Backbone frozen -> only head parameters require gradients.
    # Optimizer is built only on trainable parameters for efficiency.
    # ------------------------------------------------------------
    model = build_model(num_classes=len(dataset.classes), freeze_backbone=True).to(DEVICE)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    # ------------------------------------------------------------
    # Training loop:
    # Start with 1 epoch. If outputs look healthy, increase to 3 later.
    # ------------------------------------------------------------
    t0 = time()

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss, acc, macro_f1, cm = evaluate(model, val_loader)

        # Output format mirrors baseline so comparison is easy
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"Train avg loss: {train_loss:.4f}")
        print(f"Val avg loss:   {val_loss:.4f}")
        print(f"Accuracy:       {acc:.4f}")
        print(f"Macro F1:       {macro_f1:.4f}")
        print("\nConfusion matrix (rows=true, cols=pred):")
        print(cm)

    print(f"\nElapsed (s): {time() - t0:.1f}")

    # ------------------------------------------------------------
    # Reflection hint for Logbook v2 (you write this in the document):
    # Compare macro F1 and confusion matrix shape against the baseline.
    # The goal is to see less class collapse and better balanced predictions.
    # ------------------------------------------------------------


if __name__ == "__main__":
    main()
