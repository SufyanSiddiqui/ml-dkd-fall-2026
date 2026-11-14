"""CIFAR-100 loader and the preprocessing used by the DKD CIFAR recipe.

The official mdistiller repo is a reference for the recipe only.
This file is our own loader. It reads the public CIFAR-100 Python
batches (Krizhevsky, 2009) and applies the same train/test transforms
as mdistiller/dataset/cifar100.py:

    train: RandomCrop(32, padding=4), RandomHorizontalFlip,
           scale to [0, 1], Normalize(mean, std)
    test:  scale to [0, 1], Normalize(mean, std)

Mean and std are the CIFAR-100 constants used by that recipe, not
numbers we fitted on a subset.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from PIL import Image

# mdistiller CIFAR-100 normalization. Channel order is RGB.
CIFAR100_MEAN = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32)
CIFAR100_STD = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32)

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = _ROOT / "data" / "cifar-100-python"


def _unpickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="bytes")


def _decode_names(raw_names: list) -> list[str]:
    names = []
    for name in raw_names:
        if isinstance(name, bytes):
            names.append(name.decode("utf-8"))
        else:
            names.append(str(name))
    return names


def load_cifar100(data_dir: Path | str = DEFAULT_DATA_DIR) -> dict:
    """Load the official train and test splits.

    Returns uint8 images shaped (N, 32, 32, 3) plus fine and coarse labels.
    Nothing is normalized here. Preprocessing is a separate step so EDA can
    look at the raw pixels.
    """
    root = Path(data_dir)
    train_blob = _unpickle(root / "train")
    test_blob = _unpickle(root / "test")
    meta = _unpickle(root / "meta")

    def _images(blob: dict) -> np.ndarray:
        # Stored as (N, 3072) in channel-major order: R plane, G plane, B plane.
        flat = np.asarray(blob[b"data"], dtype=np.uint8)
        return flat.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    return {
        "train_images": _images(train_blob),
        "train_fine": np.asarray(train_blob[b"fine_labels"], dtype=np.int64),
        "train_coarse": np.asarray(train_blob[b"coarse_labels"], dtype=np.int64),
        "test_images": _images(test_blob),
        "test_fine": np.asarray(test_blob[b"fine_labels"], dtype=np.int64),
        "test_coarse": np.asarray(test_blob[b"coarse_labels"], dtype=np.int64),
        "fine_names": _decode_names(meta[b"fine_label_names"]),
        "coarse_names": _decode_names(meta[b"coarse_label_names"]),
    }


def to_float(image: np.ndarray) -> np.ndarray:
    """Match ToTensor: uint8 HWC -> float32 HWC in [0, 1]."""
    return image.astype(np.float32) / 255.0


def normalize(image: np.ndarray) -> np.ndarray:
    """Match torchvision Normalize after ToTensor. Image is float HWC in [0, 1]."""
    return (image - CIFAR100_MEAN) / CIFAR100_STD


def pad_and_random_crop(image: np.ndarray, pad: int = 4, rng: np.random.Generator | None = None) -> np.ndarray:
    """RandomCrop(32, padding=4). Pad with zeros, then take a 32x32 window.

    torchvision's default fill is 0, so the border is black. That is what
    the DKD CIFAR training loop sees. We do not reflect or replicate.
    """
    if rng is None:
        rng = np.random.default_rng()
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="constant")
    top = int(rng.integers(0, 2 * pad + 1))
    left = int(rng.integers(0, 2 * pad + 1))
    return padded[top : top + 32, left : left + 32]


def maybe_flip(image: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """RandomHorizontalFlip with p=0.5. Vertical flip is not in the recipe."""
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() < 0.5:
        return np.ascontiguousarray(image[:, ::-1, :])
    return image


def preprocess_train(image: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """One training view of a raw uint8 image. Output is float32 CHW, normalized."""
    if rng is None:
        rng = np.random.default_rng()
    view = pad_and_random_crop(image, pad=4, rng=rng)
    view = maybe_flip(view, rng=rng)
    view = normalize(to_float(view))
    return np.transpose(view, (2, 0, 1))


def preprocess_eval(image: np.ndarray) -> np.ndarray:
    """Test/val view. No crop and no flip. Output is float32 CHW, normalized."""
    view = normalize(to_float(image))
    return np.transpose(view, (2, 0, 1))


def make_splits(data: dict) -> dict:
    """Counts and per-class histograms used by the EDA and the notebook."""
    fine_names = data["fine_names"]
    coarse_names = data["coarse_names"]
    n_fine = len(fine_names)
    n_coarse = len(coarse_names)

    train_fine_counts = np.bincount(data["train_fine"], minlength=n_fine)
    test_fine_counts = np.bincount(data["test_fine"], minlength=n_fine)
    train_coarse_counts = np.bincount(data["train_coarse"], minlength=n_coarse)
    test_coarse_counts = np.bincount(data["test_coarse"], minlength=n_coarse)

    # Which fine labels sit inside each coarse label, from the training set.
    groups: dict[int, list[int]] = {i: [] for i in range(n_coarse)}
    for fine_id, coarse_id in zip(data["train_fine"].tolist(), data["train_coarse"].tolist()):
        if fine_id not in groups[coarse_id]:
            groups[coarse_id].append(fine_id)

    return {
        "n_train": int(data["train_images"].shape[0]),
        "n_test": int(data["test_images"].shape[0]),
        "n_fine": n_fine,
        "n_coarse": n_coarse,
        "image_shape": tuple(data["train_images"].shape[1:]),
        "train_fine_counts": train_fine_counts,
        "test_fine_counts": test_fine_counts,
        "train_coarse_counts": train_coarse_counts,
        "test_coarse_counts": test_coarse_counts,
        "fine_per_coarse": groups,
        "train_balanced": bool(train_fine_counts.min() == train_fine_counts.max()),
        "test_balanced": bool(test_fine_counts.min() == test_fine_counts.max()),
    }


def preprocess_pil_eval(pil_image: Image.Image) -> np.ndarray:
    """Resize/convert an arbitrary PIL image to 32x32 RGB and apply eval normalization.

    Used by the Streamlit inference app so uploaded images go through the exact
    same normalization as CIFAR-100 evaluation images.
    """
    rgb = pil_image.convert("RGB")
    if rgb.size != (32, 32):
        rgb = rgb.resize((32, 32), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.uint8)
    return preprocess_eval(arr)


class CIFAR100Dataset:
    """PyTorch-compatible Dataset wrapping our from-scratch CIFAR-100 arrays and transforms."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        train: bool = True,
        seed: int | None = None,
    ) -> None:
        if images.ndim != 4 or images.shape[1:] != (32, 32, 3):
            raise ValueError(f"Expected images of shape (N, 32, 32, 3), got {images.shape}")
        if len(images) != len(labels):
            raise ValueError(f"Mismatched lengths: {len(images)} images vs {len(labels)} labels")
        self.images = images
        self.labels = np.asarray(labels, dtype=np.int64)
        self.train = bool(train)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> tuple["torch.Tensor", int]:
        import torch

        raw = self.images[index]
        if self.train:
            chw = preprocess_train(raw, rng=self.rng)
        else:
            chw = preprocess_eval(raw)
        tensor = torch.from_numpy(np.ascontiguousarray(chw, dtype=np.float32))
        label = int(self.labels[index])
        return tensor, label


def build_dataloaders(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    batch_size: int = 64,
    eval_batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int | None = None,
) -> tuple["torch.utils.data.DataLoader", "torch.utils.data.DataLoader", dict]:
    """Build PyTorch train and test DataLoaders from the local CIFAR-100 files."""
    import torch
    from torch.utils.data import DataLoader

    raw_data = load_cifar100(data_dir)
    train_dataset = CIFAR100Dataset(
        raw_data["train_images"],
        raw_data["train_fine"],
        train=True,
        seed=seed,
    )
    test_dataset = CIFAR100Dataset(
        raw_data["test_images"],
        raw_data["test_fine"],
        train=False,
    )

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader, raw_data

