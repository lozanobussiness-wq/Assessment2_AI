import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    Simple CNN baseline for 4-class MRI classification.
    Input:  (B, 3, 224, 224)
    Output: (B, 4) logits
    """

    def __init__(self, num_classes=4):
        super().__init__()

        # Feature extractor: 3 conv blocks with progressive channel expansion.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 224 -> 112

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 112 -> 56

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 56 -> 28
        )

        # Makes the classifier robust to feature map spatial size.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def main():
    # Local sanity check: instantiate model and print parameter count.
    model = BaselineCNN(num_classes=4)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=== BaselineCNN summary ===")
    print(model)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")


if __name__ == "__main__":
    main()
