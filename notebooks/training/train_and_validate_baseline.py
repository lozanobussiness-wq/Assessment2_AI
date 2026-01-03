from time import time

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from baseline_cnn import BaselineCNN
from pipeline_utils import get_loaders

# Baseline run settings (kept local to the script for quick iteration)
BATCH_SIZE = 32
LEARNING_RATE = 1e-3


def train_one_epoch(model, loader, criterion, optimizer, device):
    # Enable training mode (important if later we add dropout/batchnorm).
    model.train()

    running_loss = 0.0
    batches = 0

    # Iterate over training batches produced by the shared data pipeline.
    for x, y in loader:
        # Move tensors to the selected device (CPU for this project).
        x, y = x.to(device), y.to(device)

        # Forward pass: compute logits for the batch.
        logits = model(x)

        # Loss uses logits directly (CrossEntropyLoss applies softmax internally).
        loss = criterion(logits, y)

        # Standard optimisation step: clear grads -> backprop -> update weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track mean loss across the epoch for a simple training signal.
        running_loss += loss.item()
        batches += 1

    # Return epoch-average loss (helps confirm learning is happening).
    return running_loss / max(1, batches)


def evaluate(model, loader, device):
    # Evaluation mode disables training-only behaviour and improves determinism.
    model.eval()

    y_true = []
    y_pred = []

    # No gradients needed during evaluation (faster + lower memory).
    with torch.no_grad():
        # Iterate over validation batches.
        for x, y in loader:
            # Only inputs need to be on device to compute logits.
            logits = model(x.to(device))

            # Convert logits to predicted class ids.
            preds = torch.argmax(logits, dim=1).cpu().tolist()

            # Collect predictions and ground-truth labels for metrics.
            y_pred.extend(preds)
            y_true.extend(y.tolist())

    # Compute baseline metrics requested by the assessment.
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)

    return acc, f1, cm


def main():
    # Load dataset + train/val loaders via shared utilities to ensure consistency.
    dataset, train_subset, val_subset, train_loader, val_loader = get_loaders(
        batch_size=BATCH_SIZE,
        num_workers=0  # Windows-friendly CPU setting
    )

    # Project constraint: train on CPU (no CUDA).
    device = torch.device("cpu")

    # Initialise the baseline CNN with 4 output classes.
    model = BaselineCNN(num_classes=4).to(device)

    # Loss for multi-class classification with integer labels.
    criterion = nn.CrossEntropyLoss()

    # Optimiser choice: simple and stable baseline on CPU.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Basic run header for reproducibility and interpretation of results.
    print("=== Baseline training + validation ===")
    print(f"Train: {len(train_subset)} | Val: {len(val_subset)}")
    print(f"Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    print(f"Classes (label order): {dataset.classes}")

    # Timing helps compare baseline vs transfer learning later.
    start = time()

    # One epoch is enough here to produce a measurable baseline without long CPU runs.
    avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

    # Evaluate on validation without touching the test set.
    acc, f1, cm = evaluate(model, val_loader, device)

    elapsed = time() - start

    # Print metrics in a report-friendly format.
    print("\n=== Validation results ===")
    print(f"Train avg loss (1 epoch): {avg_loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")

    # Confusion matrix supports per-class analysis beyond aggregate metrics.
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)

    print(f"\nElapsed (s): {elapsed:.1f}")


if __name__ == "__main__":
    main()
