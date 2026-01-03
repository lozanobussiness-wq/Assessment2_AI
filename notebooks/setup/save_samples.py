from pathlib import Path
from PIL import Image

TRAIN_DIR = Path("data/train")
RESULTS_DIR = Path("results")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Ensure results folder exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class_folders = []
for item_path in TRAIN_DIR.iterdir():
    if item_path.is_dir():
        class_folders.append(item_path)

class_folders.sort(key=str)

for class_dir in class_folders:
    image_path = None

    for file_path in class_dir.iterdir():
        if file_path.suffix.lower() in IMAGE_EXTS:
            image_path = file_path
            break

    if image_path is None:
        print(f"[WARN] No image files found in class folder: {class_dir.name}")
        continue

    img = Image.open(image_path)

    # Save a copy as PNG for easy viewing and report use
    safe_name = class_dir.name.replace(" ", "_")
    out_path = RESULTS_DIR / f"sample_{safe_name}.png"
    img.save(out_path)

    print(f"[OK] Saved: {out_path}")
