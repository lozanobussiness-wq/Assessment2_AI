import sys
from pathlib import Path
from time import time

import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ------------------------------------------------------------
# Path setup: reuse pipeline_utils from notebooks/training
# ------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[1]))
from pipeline_utils import get_loaders

# ------------------------------------------------------------
# NEW: Checkpoint paths
# - We load the head-trained model (from Phase 4 head training)
# - We save the best fine-tuned model (for Phase 5 test evaluation)
# ------------------------------------------------------------
CKPT_DIR = Path("results/transfer_learning/checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

HEAD_CKPT_PATH = CKPT_DIR / "resnet18_head_trained.pt"
BEST_CKPT_PATH = CKPT_DIR / "resnet18_finetuned_best.pt"


# ------------------------------------------------------------
# Run settings (fine-tuning uses lower LR)
# ------------------------------------------------------------
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 2
DEVICE = "cpu"


# ------------------------------------------------------------
# Build ResNet18 pretrained + new 4-class head
# Freeze all backbone, then unfreeze layer4 for partial fine-tune
#
# Note:
# - We still start from ImageNet pretrained weights,
# - but for a proper pipeline we will LOAD the head-trained checkpoint
#   right after building the model (see main()).
# ------------------------------------------------------------
def build_resnet18(num_classes):
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    # Replace final layer (fc) to output our classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze classifier head
    for p in model.fc.parameters():
        p.requires_grad = True

    # Unfreeze last residual block group (layer4)
    for p in model.layer4.parameters():
        p.requires_grad = True

    return model


# ------------------------------------------------------------
# Train for one epoch
# ------------------------------------------------------------
def train_one_epoch(model, loader, optimizer):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


# ------------------------------------------------------------
# Evaluate on validation set
# ------------------------------------------------------------
def evaluate(model, loader):
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

    return avg_loss, acc, macro_f1, cm


def main():
    print("=== Fine-tuning: ResNet18 (unfreeze layer4 + head) ===")

    # Use ImageNet normalisation for pretrained models
    dataset, train_subset, val_subset, train_loader, val_loader = get_loaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
        transform_mode="imagenet"
    )

    print(f"Train: {len(train_subset)} | Val: {len(val_subset)}")
    print(f"Batch size: {BATCH_SIZE} | LR: {LR} | Epochs: {EPOCHS}")
    print("Classes (label order):", dataset.classes)

    # Build model with the right head shape (4 classes)
    model = build_resnet18(num_classes=len(dataset.classes)).to(DEVICE)

    # ------------------------------------------------------------
    # NEW: Load head-trained checkpoint
    #
    # Why this is necessary:
    # - The head training stage produced a model where ONLY fc was trained.
    # - Fine-tuning should CONTINUE from that state, not from a fresh random fc.
    # - This ensures Phase 4 results reflect a true pipeline:
    #   head training -> fine-tuning -> test evaluation (Phase 5)
    # ------------------------------------------------------------
    if not HEAD_CKPT_PATH.exists():
        raise FileNotFoundError(
            f"Missing head checkpoint: {HEAD_CKPT_PATH}. "
            "Run train_resnet18_head.py first to generate it."
        )

    ckpt = torch.load(HEAD_CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"[CKPT] Loaded head-trained checkpoint <- {HEAD_CKPT_PATH}")

    # Sanity check: how many params train?
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params total: {total_params:,} | trainable: {trainable_params:,}")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    # ------------------------------------------------------------
    # NEW: Track the best model by validation Macro-F1
    #
    # Why Macro-F1:
    # - Your problem is 4-class classification and likely class-imbalanced.
    # - Macro-F1 weights each class equally, so it penalizes ignoring minority classes.
    # - This makes "best model" selection more robust than accuracy alone.
    # ------------------------------------------------------------
    best_val_macro_f1 = -1.0

    t0 = time()

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss, acc, macro_f1, cm = evaluate(model, val_loader)

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"Train avg loss: {train_loss:.4f}")
        print(f"Val avg loss:   {val_loss:.4f}")
        print(f"Accuracy:       {acc:.4f}")
        print(f"Macro F1:       {macro_f1:.4f}")
        print("\nConfusion matrix (rows=true, cols=pred):")
        print(cm)

        # ------------------------------------------------------------
        # NEW: Save best checkpoint (fine-tuned model) based on val Macro-F1
        # This will be the *single* model we evaluate on test in Phase 5.
        # ------------------------------------------------------------
        if macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = macro_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": dataset.class_to_idx,
                    "classes": dataset.classes,
                    "epoch": epoch,
                    "best_val_macro_f1": float(best_val_macro_f1),
                    "lr": LR,
                    "batch_size": BATCH_SIZE,
                    "stage": "fine_tuning_layer4_plus_head",
                    "source_head_checkpoint": str(HEAD_CKPT_PATH),
                },
                BEST_CKPT_PATH
            )
            print(
                f"[CKPT] Saved BEST fine-tuned checkpoint (epoch {epoch}, "
                f"macroF1={best_val_macro_f1:.4f}) -> {BEST_CKPT_PATH}"
            )

    print(f"\nElapsed (s): {time() - t0:.1f}")


if __name__ == "__main__":
    main()
