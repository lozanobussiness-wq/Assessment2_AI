from pathlib import Path

from torch.utils.data import Subset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit

# Shared configuration for the data pipeline (kept central to avoid drift)
IMAGE_SIZE = 224
SEED = 42
VAL_RATIO = 0.20

# Dataset root directory (ImageFolder expects class subfolders under this path)
TRAIN_DIR = Path("data/train")


def get_transform():
    # Standardise input size to match the expected model interface (224x224).
    # Convert grayscale MRI images to 3 channels for compatibility with ImageNet-based models.
    # Convert PIL image to float tensor in [0, 1] for PyTorch models.
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])


def get_dataset():
    # Build a single ImageFolder dataset so class mapping stays consistent everywhere.
    # This mapping (dataset.classes / class_to_idx) is used later for metrics and confusion matrix.
    return ImageFolder(root=str(TRAIN_DIR), transform=get_transform())


def get_split_indices(dataset):
    # Create a reproducible stratified split to keep class balance in train and validation.
    # We split by indices (not by copying files) to keep the dataset lightweight.
    indices = list(range(len(dataset)))
    targets = dataset.targets

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=VAL_RATIO,
        random_state=SEED
    )

    # Returns two index arrays: train_idx, val_idx
    return next(splitter.split(indices, targets))


def get_loaders(batch_size, num_workers=0):
    # Build a single ImageFolder dataset so class mapping stays consistent everywhere.
    dataset = get_dataset()

    # Create reproducible train/validation indices (stratified split).
    train_idx, val_idx = get_split_indices(dataset)

    # Wrap the same dataset with different index lists (no file copying / no duplication).
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    # Training loader: shuffle to reduce ordering bias during optimisation.
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # Validation loader: deterministic order (no shuffle) for stable evaluation.
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # Return dataset for class names/label order plus subsets/loaders for training scripts.
    return dataset, train_subset, val_subset, train_loader, val_loader
