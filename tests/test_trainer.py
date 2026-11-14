"""Unit tests for the training, evaluation, and latency benchmarking engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.cifar100 import CIFAR100Dataset
from src.engine.trainer import (
    EpochMetrics,
    adjust_learning_rate,
    append_metrics_jsonl,
    benchmark_batch_latency,
    build_optimizer,
    compute_accuracy,
    evaluate,
    load_config,
    load_resume_checkpoint,
    save_checkpoint,
     train_one_epoch,
)
from src.losses.dkd import DistillationLoss
from src.models.resnet import build_model, load_checkpoint_state_dict


def _make_tiny_loader(n_samples: int = 16, batch_size: int = 8) -> DataLoader:
    rng = np.random.default_rng(42)
    images = rng.integers(0, 256, size=(n_samples, 32, 32, 3), dtype=np.uint8)
    labels = rng.integers(0, 100, size=(n_samples,), dtype=np.int64)
    dataset = CIFAR100Dataset(images, labels, train=True, seed=42)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_config_and_lr_schedule() -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "cifar100_resnet32x4_resnet8x4.yaml"
    cfg = load_config(cfg_path)
    assert cfg["teacher"] == "resnet32x4"
    assert cfg["student"] == "resnet8x4"
    assert cfg["solver"]["epochs"] == 240
    assert cfg["dkd"]["beta"] == 8.0

    student = build_model("resnet8x4")
    opt = build_optimizer(student, lr=0.05)

    assert adjust_learning_rate(opt, epoch=1) == pytest.approx(0.05)
    assert adjust_learning_rate(opt, epoch=149) == pytest.approx(0.05)
    assert adjust_learning_rate(opt, epoch=150) == pytest.approx(0.005)
    assert adjust_learning_rate(opt, epoch=180) == pytest.approx(0.0005)
    assert adjust_learning_rate(opt, epoch=210) == pytest.approx(0.00005)


def test_compute_accuracy() -> None:
    logits = torch.tensor(
        [
            [10.0, 2.0, 1.0, 0.0, -1.0],
            [0.0, 5.0, 9.0, 1.0, -1.0],
        ]
    )
    targets = torch.tensor([0, 1])  # first is top-1, second is top-2
    top1, top2 = compute_accuracy(logits, targets, topk=(1, 2))
    assert top1 == pytest.approx(50.0)
    assert top2 == pytest.approx(100.0)


@pytest.mark.parametrize("mode", ["ce", "kd", "dkd", "tckd_only", "nckd_only"])
def test_train_one_epoch_and_evaluate_all_modes(mode: str, tmp_path: Path) -> None:
    device = torch.device("cpu")
    loader = _make_tiny_loader(n_samples=16, batch_size=8)

    student = build_model("resnet8x4", num_classes=100).to(device)
    teacher = None if mode == "ce" else build_model("resnet32x4", num_classes=100).to(device)

    criterion = DistillationLoss(mode=mode)  # type: ignore[arg-type]
    optimizer = build_optimizer(student, lr=0.01)

    train_stats = train_one_epoch(
        student=student,
        teacher=teacher,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=loader,
        device=device,
        epoch=5,
        log_interval=0,
    )
    assert train_stats["loss"] > 0.0
    assert train_stats["avg_batch_ms"] > 0.0
    if mode == "ce":
        assert train_stats["loss_distill"] == 0.0
    else:
        assert train_stats["loss_distill"] > 0.0

    eval_stats = evaluate(student, loader, device=device)
    assert eval_stats["samples"] == 16
    assert 0.0 <= eval_stats["top1"] <= 100.0

    ckpt = save_checkpoint(
        output_dir=tmp_path,
        filename="ckpt_best.pth",
        student=student,
        optimizer=optimizer,
        epoch=3,
        best_top1=eval_stats["top1"],
        mode=mode,
        config={"mode": mode},
    )
    loaded = load_checkpoint_state_dict(ckpt)
    assert "fc.weight" in loaded

    resumed_student = build_model("resnet8x4", num_classes=100)
    resumed_opt = build_optimizer(resumed_student, lr=0.01)
    next_epoch, best_acc = load_resume_checkpoint(ckpt, resumed_student, resumed_opt)
    assert next_epoch == 4
    assert best_acc == pytest.approx(eval_stats["top1"])

    metrics_file = append_metrics_jsonl(
        tmp_path,
        EpochMetrics(
            epoch=1,
            lr=0.01,
            train_loss=train_stats["loss"],
            train_loss_ce=train_stats["loss_ce"],
            train_loss_distill=train_stats["loss_distill"],
            train_top1=train_stats["top1"],
            test_loss=eval_stats["loss"],
            test_top1=eval_stats["top1"],
            test_top5=eval_stats["top5"],
            avg_batch_ms=train_stats["avg_batch_ms"],
            epoch_sec=0.5,
        ),
    )
    assert metrics_file.exists()


def test_benchmark_batch_latency_smoke() -> None:
    student = build_model("resnet8x4", num_classes=100)
    teacher = build_model("resnet32x4", num_classes=100)
    timings = benchmark_batch_latency(
        student=student,
        teacher=teacher,
        modes=("ce", "kd", "dkd"),
        batch_size=8,
        warmup_steps=1,
        timed_steps=2,
        device="cpu",
    )
    assert set(timings.keys()) == {"ce", "kd", "dkd"}
    assert all(ms > 0.0 for ms in timings.values())
