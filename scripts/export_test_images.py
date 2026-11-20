"""Export unseen CIFAR-100 test images into real .png / .jpg files on disk for live demo uploads.

Usage:
    # Default: exports a quick-pick showcase gallery + 5 images for every one of the 100 classes
    .venv/bin/python scripts/export_test_images.py

    # Optional: export ALL 10,000 unseen test images (100 per class)
    .venv/bin/python scripts/export_test_images.py --per-class 100
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CIFAR_DIR = PROJECT_ROOT / "data" / "cifar-100-python"
DEFAULT_OUT_DIR = PROJECT_ROOT / "demo_test_images"

# Visually recognizable classes placed in a flat top-level '00_quick_demo_picks/' folder
# so you can drag-and-drop during your live presentation in 2 seconds without digging through 100 folders!
SHOWCASE_CLASSES = [
    "apple",
    "bear",
    "bicycle",
    "bus",
    "cattle",
    "crocodile",
    "dolphin",
    "elephant",
    "lion",
    "motorcycle",
    "pickup_truck",
    "rocket",
    "sunflower",
    "tiger",
    "train",
]


def unpickle_file(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f, encoding="bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CIFAR-100 test split to real image files.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output folder for exported images.")
    parser.add_argument("--per-class", type=int, default=100, help="How many test images to export per class (max 100).")
    parser.add_argument("--format", choices=["png", "jpg"], default="png", help="Image file format (lossless png recommended).")
    args = parser.parse_args()

    raw_test = unpickle_file(CIFAR_DIR / "test")
    raw_meta = unpickle_file(CIFAR_DIR / "meta")

    # Exact same byte unpacking as Cell 5 of 05_dkd_from_scratch.ipynb:
    test_images = raw_test[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_labels = np.array(raw_test[b"fine_labels"], dtype=np.int64)
    fine_names = [x.decode("utf-8") for x in raw_meta[b"fine_label_names"]]

    by_class_dir = args.out_dir / "by_class"
    quick_dir = args.out_dir / "00_quick_demo_picks"
    by_class_dir.mkdir(parents=True, exist_ok=True)
    quick_dir.mkdir(parents=True, exist_ok=True)

    save_kwargs = {"quality": 100, "subsampling": 0} if args.format == "jpg" else {}

    total_saved = 0
    for class_id, class_name in enumerate(fine_names):
        class_folder = by_class_dir / class_name
        class_folder.mkdir(parents=True, exist_ok=True)

        matching_idxs = np.where(test_labels == class_id)[0][: args.per_class]
        for rank, test_idx in enumerate(matching_idxs, start=1):
            pil_img = Image.fromarray(test_images[test_idx])
            filename = f"{class_name}_{rank:02d}_testidx{test_idx:04d}.{args.format}"
            pil_img.save(class_folder / filename, **save_kwargs)
            total_saved += 1

            # Also put the first 2 images of each showcase class into the flat quick-pick folder
            if class_name in SHOWCASE_CLASSES and rank <= 2:
                pil_img.save(quick_dir / filename, **save_kwargs)

    print(f"Exported {total_saved} unseen test .{args.format} images into: {by_class_dir}")
    print(f"Exported {len(list(quick_dir.iterdir()))} presentation quick-picks into: {quick_dir}")


if __name__ == "__main__":
    main()
