from pathlib import Path

import torch
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit

from baseline_cnn import BaselineCNN

# Dataset root used throughout the project.
TRAIN_DIR = Path("data/train")

# Fixed configuration values to ensure reproducibility and consistency.
IMAGE_SIZE = 224
SEED = 42
VAL_RATIO = 0.20
BATCH_SIZE = 32


def main():
    # Input preprocessing pipeline.
    # This must remain consistent with the data pipeline validated earlier,
    # otherwise the model would be tested on a different input distribution.
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    # Load the full training dataset.
    # Class labels and their ordering are inherited from ImageFolder.
    dataset = ImageFolder(root=str(TRAIN_DIR), transform=transform)

    # Build a reproducible, stratified split.
    # Only the training subset is needed here, since we are not validating yet.
    indices = list(range(len(dataset)))
    targets = dataset.targets

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=VAL_RATIO,
        random_state=SEED
    )
    train_idx, _ = next(splitter.split(indices, targets))

    # Subset applies the split by index without duplicating data.
    train_subset = Subset(dataset, train_idx)

    # DataLoader handles batching and shuffling.
    # Shuffling is enabled to reflect the conditions used during training.
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0  # CPU-only, Windows-stable configuration
    )

    # Explicitly use CPU to match the project constraints.
    device = torch.device("cpu")

    # Instantiate the baseline model and switch to evaluation mode.
    # eval() disables training-specific layers such as dropout (if present).
    model = BaselineCNN(num_classes=4).to(device)
    model.eval()

    # Retrieve a single batch from the DataLoader.
    # This represents real data flowing through the entire pipeline.
    x_batch, y_batch = next(iter(train_loader))
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    # Forward pass only: no gradients, no optimisation.
    # This validates tensor compatibility and output shape.
    with torch.no_grad():
        logits = model(x_batch)

    print("=== Forward pass check ===")
    print(f"Input batch shape:  {tuple(x_batch.shape)}  (B, C, H, W)")
    print(f"Labels shape:       {tuple(y_batch.shape)}  (B,)")
    print(f"Logits shape:       {tuple(logits.shape)}  (B, num_classes)")
    print(f"Logits dtype:       {logits.dtype}")

    # Numerical sanity check: logits must be finite values.
    # NaNs or infinities here would indicate serious issues upstream.
    print(f"Logits min/max:     {logits.min().item():.4f} / {logits.max().item():.4f}")
    print(f"All logits finite:  {torch.isfinite(logits).all().item()}")

    # Inspect a few predictions to confirm output indexing,
    # without making any claims about model performance.
    preds = torch.argmax(logits, dim=1)
    print("\nFirst 10 predictions (label ids):", preds[:10].tolist())
    print("First 10 true labels (label ids):", y_batch[:10].tolist())


if __name__ == "__main__":
    main()
