from pathlib import Path
from collections import Counter

from torchvision.datasets import ImageFolder


def summarize_split(split_path):
    dataset = ImageFolder(root=str(split_path))
    class_counts = Counter(dataset.targets)

    return dataset.classes, class_counts, len(dataset)


def main():
    train_dir = Path("data/train")
    test_dir = Path("data/test")

    print("=== Dataset summary ===\n")

    # Training split (before internal train/val split)
    train_classes, train_counts, train_total = summarize_split(train_dir)

    print("Train split:")
    print(f"Total images: {train_total}")
    for idx, class_name in enumerate(train_classes):
        print(f"  {class_name}: {train_counts[idx]}")
    print()

    # Test split
    test_classes, test_counts, test_total = summarize_split(test_dir)

    print("Test split:")
    print(f"Total images: {test_total}")
    for idx, class_name in enumerate(test_classes):
        print(f"  {class_name}: {test_counts[idx]}")
    print()


if __name__ == "__main__":
    main()
