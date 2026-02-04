from pathlib import Path
from time import time

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


# ------------------------------------------------------------
# Settings (keep consistent with project)
# ------------------------------------------------------------
BATCH_SIZE = 32
DEVICE = "cpu"
NUM_WORKERS = 0  # Windows-safe


# ------------------------------------------------------------
# Paths (project-relative)
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # /ASSESSMENT2_AI
TEST_DIR = PROJECT_ROOT / "data" / "test"
CKPT_PATH = PROJECT_ROOT / "results" / "transfer_learning" / "checkpoints" / "resnet18_finetuned_best.pt"


# ------------------------------------------------------------
# Eval transforms (no augmentation)
# ------------------------------------------------------------
def get_test_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ------------------------------------------------------------
# Model must match training: ResNet18 + custom fc
# ------------------------------------------------------------
def build_resnet18(num_classes: int) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# ------------------------------------------------------------
# Pure evaluation loop (no training)
# ------------------------------------------------------------
def evaluate(model: nn.Module, loader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, acc, macro_f1, cm, all_labels, all_preds


def main():
    print("=== Phase 5: Final Test Evaluation (ResNet18 fine-tuned) ===")

    # --- Safety checks ---
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CKPT_PATH}\n"
            "Expected: resnet18_finetuned_best.pt from fine_tune_resnet18_last3.py."
        )

    # Load checkpoint to verify class order consistency
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    ckpt_classes = ckpt.get("classes", None)

    # Build test dataset/loader
    test_dataset = ImageFolder(str(TEST_DIR), transform=get_test_transform())
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,          # deterministic
        num_workers=NUM_WORKERS
    )

    print(f"Test samples: {len(test_dataset)}")
    print("Test classes (folder order):", test_dataset.classes)

    # Ensure test label order matches training checkpoint label order
    if ckpt_classes is not None and list(test_dataset.classes) != list(ckpt_classes):
        raise ValueError(
            "Class order mismatch between checkpoint and test dataset.\n"
            f"Checkpoint classes: {ckpt_classes}\n"
            f"Test dataset classes: {test_dataset.classes}\n"
            "Fix this before evaluating test, otherwise metrics will be invalid."
        )

    # Build model + load weights
    model = build_resnet18(num_classes=len(test_dataset.classes)).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    print(f"[CKPT] Loaded: {CKPT_PATH}")
    print(f"Batch size: {BATCH_SIZE} | Device: {DEVICE}")

    # Run evaluation once
    t0 = time()
    test_loss, acc, macro_f1, cm, y_true, y_pred = evaluate(model, test_loader)
    elapsed = time() - t0

    print("\n--- TEST RESULTS (FINAL) ---")
    print(f"Test avg loss: {test_loss:.4f}")
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test macro F1: {macro_f1:.4f}")
    print(f"Elapsed (s): {elapsed:.1f}")

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)

    print("\nClassification report (per class):")
    print(classification_report(y_true, y_pred, target_names=test_dataset.classes, digits=4))


if __name__ == "__main__":
    main()
