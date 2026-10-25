"""Exploratory stats and figures for the CIFAR-100 mid-evaluation report.

Run from the repo root after data/cifar-100-python exists:

    .venv/bin/python scripts/eda_cifar100.py

Figures land in docs/mid_eval/figures/. A small JSON summary is written next
to them so the report numbers come from this run, not from memory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.cifar100 import (  # noqa: E402
    CIFAR100_MEAN,
    CIFAR100_STD,
    load_cifar100,
    make_splits,
    preprocess_eval,
    preprocess_train,
)

FIG_DIR = ROOT / "docs" / "mid_eval" / "figures"


def _save(fig: plt.Figure, name: str) -> None:
    path = FIG_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def class_grid(images: np.ndarray, labels: np.ndarray, names: list[str], class_ids: list[int], title: str, name: str) -> None:
    cols = len(class_ids)
    fig, axes = plt.subplots(1, cols, figsize=(2.1 * cols, 2.6))
    for ax, class_id in zip(axes, class_ids):
        idx = int(np.flatnonzero(labels == class_id)[0])
        ax.imshow(images[idx])
        ax.set_title(names[class_id].replace("_", " "), fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    _save(fig, name)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_cifar100()
    splits = make_splits(data)
    train = data["train_images"]
    test = data["test_images"]

    # Pixel mean/std on the training set, in [0, 1], to compare with the recipe.
    train_float = train.astype(np.float64) / 255.0
    computed_mean = train_float.mean(axis=(0, 1, 2))
    computed_std = train_float.std(axis=(0, 1, 2))

    # Brightness: mean over RGB, then over pixels, per image.
    brightness = train_float.mean(axis=(1, 2, 3))

    summary = {
        "n_train": splits["n_train"],
        "n_test": splits["n_test"],
        "n_fine": splits["n_fine"],
        "n_coarse": splits["n_coarse"],
        "image_shape": list(splits["image_shape"]),
        "dtype": str(train.dtype),
        "pixel_min": int(train.min()),
        "pixel_max": int(train.max()),
        "train_per_class_min": int(splits["train_fine_counts"].min()),
        "train_per_class_max": int(splits["train_fine_counts"].max()),
        "test_per_class_min": int(splits["test_fine_counts"].min()),
        "test_per_class_max": int(splits["test_fine_counts"].max()),
        "train_balanced": splits["train_balanced"],
        "test_balanced": splits["test_balanced"],
        "train_per_superclass_min": int(splits["train_coarse_counts"].min()),
        "train_per_superclass_max": int(splits["train_coarse_counts"].max()),
        "computed_mean": [round(float(x), 4) for x in computed_mean],
        "computed_std": [round(float(x), 4) for x in computed_std],
        "recipe_mean": [float(x) for x in CIFAR100_MEAN],
        "recipe_std": [float(x) for x in CIFAR100_STD],
        "brightness_p01": round(float(np.quantile(brightness, 0.01)), 4),
        "brightness_p50": round(float(np.quantile(brightness, 0.50)), 4),
        "brightness_p99": round(float(np.quantile(brightness, 0.99)), 4),
        "fine_per_coarse": {
            data["coarse_names"][k]: [data["fine_names"][i] for i in v]
            for k, v in splits["fine_per_coarse"].items()
        },
    }

    # One raw example from five fine classes that sit in different superclasses.
    # Picked after load so the ids match the official name list.
    wanted = ["apple", "beaver", "motorcycle", "oak_tree", "rocket"]
    name_to_id = {name: i for i, name in enumerate(data["fine_names"])}
    example_ids = [name_to_id[name] for name in wanted]
    class_grid(
        train,
        data["train_fine"],
        data["fine_names"],
        example_ids,
        "Raw CIFAR-100 training images (32x32)",
        "raw_examples.png",
    )

    # Same superclass, five fine classes: vehicles 1 is the usual confusable group.
    vehicles = None
    for coarse_id, fine_ids in splits["fine_per_coarse"].items():
        if data["coarse_names"][coarse_id] == "vehicles_1":
            vehicles = fine_ids
            break
    if vehicles is None:
        raise SystemExit("vehicles_1 superclass missing from the label file")
    class_grid(
        train,
        data["train_fine"],
        data["fine_names"],
        vehicles,
        "Five fine classes inside the vehicles_1 superclass",
        "superclass_vehicles.png",
    )

    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.bar(np.arange(splits["n_fine"]), splits["train_fine_counts"], color="#3d5a80", width=0.9)
    ax.set_xlabel("Fine class id (0-99)")
    ax.set_ylabel("Training images")
    ax.set_title("CIFAR-100 training set is balanced: 500 images in every fine class")
    ax.set_ylim(0, 600)
    _save(fig, "class_balance.png")

    fig, ax = plt.subplots(figsize=(10, 4.2))
    order = np.argsort(splits["train_coarse_counts"])
    ax.barh(
        [data["coarse_names"][i].replace("_", " ") for i in order],
        splits["train_coarse_counts"][order],
        color="#ee6c4d",
    )
    ax.set_xlabel("Training images")
    ax.set_title("Training images per superclass (20 groups, 5 fine classes each)")
    _save(fig, "superclass_counts.png")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(brightness, bins=40, color="#3d5a80", edgecolor="white")
    ax.set_xlabel("Mean pixel value per image, scaled to [0, 1]")
    ax.set_ylabel("Training images")
    ax.set_title("Brightness is spread out; a few images are nearly black or nearly white")
    _save(fig, "brightness.png")

    # Show that the training transform actually changes a sample, and that
    # eval does not. Use a fixed seed so the figure is reproducible.
    rng = np.random.default_rng(0)
    sample = train[0]
    views = [sample]
    for _ in range(3):
        chw = preprocess_train(sample, rng=rng)
        # Undo normalize only for display.
        hwc = np.transpose(chw, (1, 2, 0))
        hwc = hwc * CIFAR100_STD + CIFAR100_MEAN
        views.append(np.clip(hwc, 0, 1))
    eval_hwc = np.transpose(preprocess_eval(sample), (1, 2, 0))
    eval_hwc = np.clip(eval_hwc * CIFAR100_STD + CIFAR100_MEAN, 0, 1)

    fig, axes = plt.subplots(1, 5, figsize=(10, 2.5))
    titles = ["raw", "train view 1", "train view 2", "train view 3", "eval view"]
    shown = [views[0], views[1], views[2], views[3], eval_hwc]
    for ax, img, title in zip(axes, shown, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(
        f"Preprocessing on one {data['fine_names'][int(data['train_fine'][0])].replace('_', ' ')} image",
        fontsize=11,
    )
    _save(fig, "preprocess_views.png")

    out = FIG_DIR / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    print(json.dumps({k: summary[k] for k in summary if k != "fine_per_coarse"}, indent=2))


if __name__ == "__main__":
    main()
