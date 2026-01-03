from pathlib import Path
from time import time

import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit

from baseline_cnn import BaselineCNN

TRAIN_DIR = Path("data/train")

IMAGE_SIZE = 224
SEED = 42
VAL_RATIO = 0.20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3


def main():
    # Preprocessing consistent with the pipeline used so far.
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(root=str(TRAIN_DIR), transform=transform)

    # Reproducible, stratified split; only train subset is used in this script.
    indices = list(range(len(dataset)))
    targets = dataset.targets

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=SEED)
    train_idx, _ = next(splitter.split(indices, targets))

    train_subset = Subset(dataset, train_idx)
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # CPU-only training (project constraint).
    device = torch.device("cpu")

    model = BaselineCNN(num_classes=4).to(device)

    # CrossEntropyLoss expects raw logits (no softmax) and integer class labels.
    criterion = nn.CrossEntropyLoss()

    # Adam is a practical default for a baseline on CPU.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()

    start = time()
    running_loss = 0.0
    num_batches = 0

    print("=== One-epoch training sanity check ===")
    print(f"Train samples: {len(train_subset)} | Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE}")

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        # Backprop + parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

        # Lightweight progress printing (useful for CPU runs)
        if num_batches % 50 == 0:
            avg_loss = running_loss / num_batches
            print(f"Batch {num_batches:4d} | Avg loss: {avg_loss:.4f}")

    avg_epoch_loss = running_loss / max(1, num_batches)
    elapsed = time() - start

    print("\n=== Training summary ===")
    print(f"Batches processed: {num_batches}")
    print(f"Average epoch loss: {avg_epoch_loss:.4f}")
    print(f"Elapsed time (s): {elapsed:.1f}")


if __name__ == "__main__":
    main()
