from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit

# Training dataset directory (subfolders are treated as class labels)
TRAIN_DIR = Path("data/train")

# Fixed project settings (kept here to make runs reproducible and easy to audit)
IMAGE_SIZE = 224
SEED = 42
VAL_RATIO = 0.20
BATCH_SIZE = 32


def main():
    # Preprocessing pipeline to standardise inputs for CNNs and ImageNet-based models:
    # - resize to a fixed spatial size
    # - ensure 3 channels (MRI is typically grayscale)
    # - convert to float tensor in [0, 1]
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    # ImageFolder builds:
    # - class list from subfolder names
    # - integer labels assigned by alphabetical folder order
    dataset = ImageFolder(root=str(TRAIN_DIR), transform=transform)

    # Dataset indices and corresponding labels are used to create a reproducible split.
    # We keep a single dataset and split by indices (no duplication of image files).
    indices = list(range(len(dataset)))
    targets = dataset.targets

    # Stratified split maintains class proportions in both training and validation subsets.
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=VAL_RATIO,
        random_state=SEED
    )
    train_idx, val_idx = next(splitter.split(indices, targets))

    # Subset wraps the original dataset with a fixed list of indices.
    # This is how we create train/validation datasets without moving files around.
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    # DataLoader handles batching and shuffling.
    # Windows stability: num_workers=0 avoids multiprocessing edge cases on local CPU runs.
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,          # shuffle for training to reduce ordering bias
        num_workers=0,
        pin_memory=False       # CPU-only training: keep it simple and predictable
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,         # validation should be deterministic
        num_workers=0,
        pin_memory=False
    )

    # High-level loader sanity check: confirms split sizes and batch settings.
    print("=== DataLoader sanity check ===")
    print(f"Train subset: {len(train_subset)} samples")
    print(f"Val subset:   {len(val_subset)} samples")
    print(f"Batch size:   {BATCH_SIZE}")

    # Fetch one batch to verify end-to-end pipeline:
    # dataset -> subset -> dataloader -> batch tensors
    x_batch, y_batch = next(iter(train_loader))

    # Shape conventions:
    # - images: (B, C, H, W)
    # - labels: (B,)
    print("\n=== One batch check (train) ===")
    print(f"x dtype:  {x_batch.dtype}")
    print(f"x shape:  {tuple(x_batch.shape)}  (B, C, H, W)")
    print(f"y dtype:  {y_batch.dtype}")
    print(f"y shape:  {tuple(y_batch.shape)}  (B,)")

    # Basic label validation:
    # - labels must be within the expected range [0, num_classes-1]
    # - batch should contain plausible class labels (not all the same unless by chance)
    y_list = y_batch.tolist()
    label_counts = Counter(y_list)

    print("\nLabel counts in this batch (label -> count):")
    print(dict(label_counts))

    # Mapping labels back to human-readable class names helps interpret outputs.
    print("\nLabel names present in this batch:")
    for label in sorted(label_counts.keys()):
        print(f"  {label}: {dataset.classes[label]}")

    # Quick value-range check (before normalisation is introduced).
    # Useful to detect unexpected scaling or dtype issues early.
    print(f"\nBatch min/max: {x_batch.min().item():.4f} / {x_batch.max().item():.4f}")


# Script entry point (allows importing this module without running it).
if __name__ == "__main__":
    main()
