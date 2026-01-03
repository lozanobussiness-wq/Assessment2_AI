from pathlib import Path

DATA_DIR = Path("data")

def count_split(split_name: str):
    split_dir = DATA_DIR / split_name
    print(f"\n== {split_name.upper()} ==")

    if not split_dir.exists():
        print(f"NO EXISTE: {split_dir.resolve()}")
        return

    for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        count = sum(
            1 for f in cls_dir.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
        )
        print(f"{cls_dir.name}: {count}")

count_split("train")
count_split("test")
