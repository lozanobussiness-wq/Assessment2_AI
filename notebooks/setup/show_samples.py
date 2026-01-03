from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Directory containing class folders
TRAIN_DIR = Path("data/train")

# Accepted image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Collect class folders explicitly (readable and beginner-friendly)
class_folders = []
for item_path in TRAIN_DIR.iterdir():
    if item_path.is_dir():
        class_folders.append(item_path)

# Sort to keep output order consistent across runs (no lambda)
class_folders.sort(key=str)

for class_dir in class_folders:
    image_path = None

    # Find the first image file inside the class folder
    for file_path in class_dir.iterdir():
        if file_path.suffix.lower() in IMAGE_EXTS:
            image_path = file_path
            break

    # If no image file was found, warn and skip this class folder
    if image_path is None:
        print(f"[WARN] No image files found in class folder: {class_dir.name}")
        continue

    # Load the image
    img = Image.open(image_path)

    # Display the image
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.title(f"{class_dir.name} | {image_path.name} | size={img.size}")
    plt.axis("off")
    plt.show()
