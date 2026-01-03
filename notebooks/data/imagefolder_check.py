from pathlib import Path
from torchvision.datasets import ImageFolder

# Training dataset directory.
# Expected structure:
# data/train/<class_name>/*.png|jpg|...
TRAIN_DIR = Path("data/train")


def main():
    # Quick dataset sanity check before building transforms/splits/dataloaders.
    # Confirms class discovery and label assignment (used throughout training/evaluation).
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Train directory not found: {TRAIN_DIR.resolve()}")

    # ImageFolder treats each subfolder as a class and assigns labels
    # based on alphabetical folder order.
    dataset = ImageFolder(root=str(TRAIN_DIR))

    print("=== ImageFolder sanity check ===")
    print(f"Dataset root: {TRAIN_DIR.resolve()}")
    print(f"Total images found: {len(dataset)}")
    print(f"Number of classes found: {len(dataset.classes)}")

    print("\nClasses detected (index -> class name):")
    for idx, class_name in enumerate(dataset.classes):
        print(f"  {idx}: {class_name}")

    # Explicit mapping used internally (class name -> numeric label).
    print("\nClass to index mapping (class name -> label):")
    print(dataset.class_to_idx)


# Script entry point.
if __name__ == "__main__":
    main()
