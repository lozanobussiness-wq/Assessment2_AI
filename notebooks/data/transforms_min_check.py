from pathlib import Path

import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms

# Dataset root (same as Step 1)
TRAIN_DIR = Path("data/train")


def main():
    # Minimal preprocessing for a first sanity check.
    # No normalization yet: we only want to confirm shapes/types/ranges.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # ensure 3 channels for ImageNet models
        transforms.ToTensor(),  # converts to float tensor in [0, 1]
    ])

    dataset = ImageFolder(root=str(TRAIN_DIR), transform=transform)

    # Take a single sample to validate the pipeline.
    image_tensor, label = dataset[0]

    print("=== Transform sanity check (single sample) ===")
    print(f"Tensor dtype: {image_tensor.dtype}")
    print(f"Tensor shape: {tuple(image_tensor.shape)}  (C, H, W)")
    print(f"Label (int): {label}")
    print(f"Label name: {dataset.classes[label]}")

    # Value range check (useful before normalization is introduced)
    print(f"Tensor min/max: {image_tensor.min().item():.4f} / {image_tensor.max().item():.4f}")

    # Optional: confirm it's a torch Tensor
    print(f"Is torch.Tensor: {isinstance(image_tensor, torch.Tensor)}")


if __name__ == "__main__":
    main()


