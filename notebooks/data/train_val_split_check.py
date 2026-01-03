from pathlib import Path
from collections import Counter

from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit

# Same dataset root used in previous steps
TRAIN_DIR = Path("data/train")

# Fixed project choices
IMAGE_SIZE = 224
SEED = 42
VAL_RATIO = 0.20


def main():
    # Minimal transforms: needed so ImageFolder can produce consistent samples later.
    # (We don't inspect tensors here, but keep the same preprocessing path.)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(root=str(TRAIN_DIR), transform=transform)

    # Targets are the numeric labels assigned by ImageFolder
    targets = dataset.targets
    indices = list(range(len(dataset)))

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=VAL_RATIO,
        random_state=SEED
    )

    train_idx, val_idx = next(splitter.split(indices, targets))

    # Basic size checks
    print("=== Train/Validation split sanity check ===")
    print(f"Total samples: {len(dataset)}")
    print(f"Train samples: {len(train_idx)}")
    print(f"Val samples:   {len(val_idx)}")
    print(f"Val ratio:     {len(val_idx) / len(dataset):.3f}")

    # Class distribution checks (counts per label)
    train_counts = Counter([targets[i] for i in train_idx])
    val_counts = Counter([targets[i] for i in val_idx])

    print("\nClass distribution (label -> count):")
    print("Train:", dict(train_counts))
    print("Val:  ", dict(val_counts))

    # Same check, but with class names for readability
    print("\nClass distribution (class name -> count):")
    for label, class_name in enumerate(dataset.classes):
        t_count = train_counts.get(label, 0)
        v_count = val_counts.get(label, 0)
        print(f"{class_name:25s} | train={t_count:4d} | val={v_count:4d}")

    # Quick reproducibility hint (optional): print first 10 indices
    print("\nFirst 10 train indices:", list(train_idx[:10]))
    print("First 10 val indices:  ", list(val_idx[:10]))


if __name__ == "__main__":
    main()
