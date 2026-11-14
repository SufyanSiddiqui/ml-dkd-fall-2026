"""Training, evaluation, and batch-timing engine for CIFAR-100 DKD reproduction.

Follows the exact solver protocol in configs/cifar100_resnet32x4_resnet8x4.yaml:
- Optimizer: SGD(lr=0.05, momentum=0.9, weight_decay=5e-4)
- Epochs: 240
- LR schedule: decay by 0.1 at epochs [150, 180, 210]
- Warmup: linear warmup over first 20 epochs for distillation loss
- Evaluation metric: CIFAR-100 top-1 accuracy (and top-5 for diagnostics)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.losses.dkd import DistillationLoss, LossMode


@dataclass
class EpochMetrics:
    """Summary metrics recorded at the end of one training epoch."""

    epoch: int
    lr: float
    train_loss: float
    train_loss_ce: float
    train_loss_distill: float
    train_top1: float
    test_loss: float
    test_top1: float
    test_top5: float
    avg_batch_ms: float
    epoch_sec: float


def select_device(requested: str = "auto") -> torch.device:
    """Select computation device ('cuda', 'mps', or 'cpu')."""
    req = requested.lower()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(req)


def synchronize_device(device: torch.device) -> None:
    """Synchronize accelerator stream for accurate batch wall-clock timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def set_seed(seed: int) -> None:
    """Seed NumPy and PyTorch RNGs for reproducible runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: Path | str) -> dict[str, Any]:
    """Load YAML training configuration."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def compute_accuracy(
    logits: torch.Tensor,
    target: torch.Tensor,
    topk: tuple[int, ...] = (1, 5),
) -> list[ float]:
    """Compute top-k accuracy percentages (0-100) for the given batch."""
    with torch.no_grad():
        maxk = min(max(topk), logits.size(1))
        batch_size = target.size(0)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        results: list[float] = []
        for k in topk:
            k_eff = min(k, logits.size(1))
            correct_k = correct[:k_eff].reshape(-1).float().sum(0)
            results.append(float(correct_k.mul_(100.0 / batch_size).item()))
        return results


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    base_lr: float = 0.05,
    decay_stages: list[int] | tuple[int, ...] = (150, 180, 210),
    decay_rate: float = 0.1,
) -> float:
    """Step-decay learning rate matching mdistiller's CIFAR schedule (1-indexed epoch)."""
    steps = sum(1 for stage in decay_stages if epoch >= int(stage))
    lr = float(base_lr * (decay_rate**steps))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def build_optimizer(
    model: nn.Module,
    lr: float = 0.05,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
) -> torch.optim.SGD:
    """Create SGD optimizer matching configs/cifar100_resnet32x4_resnet8x4.yaml."""
    return torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate top-1, top-5 accuracy and cross-entropy loss on a DataLoader."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_samples = 0
    loss_sum = 0.0
    top1_sum = 0.0
    top5_sum = 0.0

    for batch_idx, (images, targets) in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)
        acc1, acc5 = compute_accuracy(logits, targets, topk=(1, 5))

        bs = int(targets.size(0))
        total_samples += bs
        loss_sum += float(loss.item()) * bs
        top1_sum += acc1 * bs
        top5_sum += acc5 * bs

    if total_samples == 0:
        return {"loss": 0.0, "top1": 0.0, "top5": 0.0, "samples": 0}

    return {
        "loss": loss_sum / total_samples,
        "top1": top1_sum / total_samples,
        "top5": top5_sum / total_samples,
        "samples": total_samples,
    }


def train_one_epoch(
    student: nn.Module,
    teacher: nn.Module | None,
    criterion: DistillationLoss,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device,
    epoch: int,
    max_batches: int | None = None,
    log_interval: int = 100,
) -> dict[str, float]:
    """Run one training epoch and measure per-batch step latency in milliseconds."""
    student.train()
    if teacher is not None:
        teacher.eval()

    total_samples = 0
    loss_sum = 0.0
    ce_sum = 0.0
    distill_sum = 0.0
    top1_sum = 0.0
    batch_times_ms: list[float] = []

    for batch_idx, (images, targets) in enumerate(train_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        synchronize_device(device)
        t0 = time.perf_counter()

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits_student = student(images)
        logits_teacher = None
        if criterion.mode != "ce":
            if teacher is None:
                raise ValueError(f"Teacher model is required when mode={criterion.mode!r}")
            with torch.no_grad():
                logits_teacher = teacher(images)

        loss_dict = criterion(
            logits_student=logits_student,
            target=targets,
            logits_teacher=logits_teacher,
            epoch=epoch,
        )
        loss = loss_dict["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        synchronize_device(device)
        batch_ms = (time.perf_counter() - t0) * 1000.0
        batch_times_ms.append(batch_ms)

        acc1 = compute_accuracy(logits_student, targets, topk=(1,))[0]
        bs = int(targets.size(0))
        total_samples += bs
        loss_sum += float(loss.item()) * bs
        ce_sum += float(loss_dict["loss_ce"].item()) * bs
        distill_sum += float(loss_dict["loss_distill"].item()) * bs
        top1_sum += acc1 * bs

        if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
            print(
                f"  [Epoch {epoch:03d} | Batch {batch_idx + 1:03d}/{len(train_loader)}] "
                f"loss={loss_sum / total_samples:.4f} "
                f"(ce={ce_sum / total_samples:.4f}, distill={distill_sum / total_samples:.4f}) "
                f"top1={top1_sum / total_samples:.2f}% "
                f"batch={np.mean(batch_times_ms):.2f}ms"
            )

    if total_samples == 0:
        return {
            "loss": 0.0,
            "loss_ce": 0.0,
            "loss_distill": 0.0,
            "top1": 0.0,
            "avg_batch_ms": 0.0,
        }

    return {
        "loss": loss_sum / total_samples,
        "loss_ce": ce_sum / total_samples,
        "loss_distill": distill_sum / total_samples,
        "top1": top1_sum / total_samples,
        "avg_batch_ms": float(np.mean(batch_times_ms)) if batch_times_ms else 0.0,
    }


def benchmark_batch_latency(
    student: nn.Module,
    teacher: nn.Module | None,
    modes: tuple[LossMode, ...] = ("ce", "kd", "dkd", "tckd_only", "nckd_only"),
    batch_size: int = 64,
    num_classes: int = 100,
    warmup_steps: int = 5,
    timed_steps: int = 20,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Compare average training step time (ms/batch) across loss modes under identical conditions.

    Verifies the Category-A ablation requirement that DKD adds negligible overhead over
    vanilla KD because both reuse the exact same single student + teacher forward pass.
    """
    dev = torch.device(device) if isinstance(device, str) else device
    student = student.to(dev)
    student.train()
    if teacher is not None:
        teacher = teacher.to(dev)
        teacher.eval()

    x = torch.randn(batch_size, 3, 32, 32, device=dev)
    y = torch.randint(0, num_classes, (batch_size,), device=dev)
    optimizer = build_optimizer(student)

    timings_ms: dict[str, float] = {}
    for mode in modes:
        if mode != "ce" and teacher is None:
            continue
        criterion = DistillationLoss(mode=mode)
        times: list[float] = []

        for step in range(warmup_steps + timed_steps):
            synchronize_device(dev)
            t0 = time.perf_counter()

            logits_s = student(x)
            logits_t = None
            if mode != "ce" and teacher is not None:
                with torch.no_grad():
                    logits_t = teacher(x)

            out = criterion(logits_s, y, logits_t, epoch=20)
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            optimizer.step()

            synchronize_device(dev)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if step >= warmup_steps:
                times.append(elapsed_ms)

        timings_ms[mode] = float(np.mean(times))

    return timings_ms


def save_checkpoint(
    output_dir: Path,
    filename: str,
    student: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_top1: float,
    mode: str,
    config: dict[str, Any],
) -> Path:
    """Save student checkpoint in mdistiller-compatible format {'model': state_dict, ...}."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    payload = {
        "epoch": epoch,
        "best_top1": best_top1,
        "mode": mode,
        "model": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
    }
    torch.save(payload, path)
    return path


def load_resume_checkpoint(
    checkpoint_path: Path | str,
    student: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[int, float]:
    """Restore student weights, optimizer state, last completed epoch, and best_top1.

    Returns (start_epoch, best_top1) where start_epoch is last_epoch + 1.
    Useful on Google Colab if a 240-epoch session disconnects and resumes from Google Drive.
    """
    raw = torch.load(Path(checkpoint_path), map_location=map_location, weights_only=False)
    if not isinstance(raw, dict) or "model" not in raw:
        raise ValueError(f"Expected training checkpoint dict with 'model' key at {checkpoint_path}")

    student.load_state_dict(raw["model"], strict=True)
    if optimizer is not None and "optimizer" in raw:
        optimizer.load_state_dict(raw["optimizer"])

    last_epoch = int(raw.get("epoch", 0))
    best_top1 = float(raw.get("best_top1", 0.0))
    return last_epoch + 1, best_top1


def append_metrics_jsonl(output_dir: Path, metrics: EpochMetrics) -> Path:
    """Append one epoch's metrics as a JSON line to metrics.jsonl."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "metrics.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(metrics)) + "\n")
    return log_path
