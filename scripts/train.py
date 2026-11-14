"""CLI script to train CIFAR-100 ResNet8x4 students (CE, KD, DKD, TCKD-only, NCKD-only).

Examples:
    # Full 240-epoch DKD reproduction run:
    .venv/bin/python scripts/train.py --mode dkd

    # Category-A ablation runs:
    .venv/bin/python scripts/train.py --mode ce
    .venv/bin/python scripts/train.py --mode kd
    .venv/bin/python scripts/train.py --mode tckd_only
    .venv/bin/python scripts/train.py --mode nckd_only

    # Quick smoke run (1 epoch, 2 batches) + batch-timing benchmark:
    .venv/bin/python scripts/train.py --mode dkd --epochs 1 --max-batches 2 --benchmark-timing
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.cifar100 import DEFAULT_DATA_DIR, build_dataloaders
from src.engine.trainer import (
    EpochMetrics,
    adjust_learning_rate,
    append_metrics_jsonl,
    benchmark_batch_latency,
    build_optimizer,
    evaluate,
    load_config,
    load_resume_checkpoint,
    save_checkpoint,
    select_device,
    set_seed,
    train_one_epoch,
)
from src.losses.dkd import DistillationLoss
from src.models.resnet import build_model, load_pretrained_teacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CIFAR-100 ResNet8x4 student with CE, KD, DKD, TCKD-only, or NCKD-only."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "cifar100_resnet32x4_resnet8x4.yaml",
        help="Path to the CIFAR-100 YAML config file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="dkd",
        choices=list(DistillationLoss.VALID_MODES),
        help="Training loss mode: ce | kd | dkd | tckd_only | nckd_only.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing extracted CIFAR-100 train/test/meta files.",
    )
    parser.add_argument(
        "--teacher-ckpt",
        type=Path,
        default=None,
        help="Path to pretrained ResNet32x4 teacher checkpoint (auto-downloaded if omitted).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for checkpoints and metrics.jsonl (defaults to outputs/<mode>).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs from config (default: 240).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training batch size from config (default: 64).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override DKD alpha weight (default from config: 1.0).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Override DKD beta weight (default from config: 8.0).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override distillation temperature T (default from config: 4.0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: auto | cuda | mps | cpu.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on batches per epoch (for fast smoke tests).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Print batch progress every N batches (0 to disable).",
    )
    parser.add_argument(
        "--benchmark-timing",
        action="store_true",
        help="Run a ms/batch timing comparison across CE, KD, DKD, TCKD-only, and NCKD-only before training.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from output_dir/ckpt_last.pth if present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = select_device(args.device)

    solver_cfg = cfg["solver"]
    dkd_cfg = cfg["dkd"]

    epochs = int(args.epochs if args.epochs is not None else solver_cfg["epochs"])
    batch_size = int(args.batch_size if args.batch_size is not None else solver_cfg["batch_size"])
    base_lr = float(solver_cfg["lr"])
    momentum = float(solver_cfg["momentum"])
    weight_decay = float(solver_cfg["weight_decay"])
    lr_decay_stages = [int(s) for s in solver_cfg["lr_decay_stages"]]
    lr_decay_rate = float(solver_cfg["lr_decay_rate"])

    alpha = float(args.alpha if args.alpha is not None else dkd_cfg["alpha"])
    beta = float(args.beta if args.beta is not None else dkd_cfg["beta"])
    temperature = float(args.temperature if args.temperature is not None else dkd_cfg["temperature"])
    ce_weight = float(dkd_cfg["ce_weight"])
    warmup_epochs = int(dkd_cfg["warmup_epochs"])

    output_dir = args.output_dir or (ROOT / "outputs" / args.mode)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== CIFAR-100 DKD Training ({args.mode}) ===")
    print(f"Device:      {device}")
    print(f"Student:     {cfg['student']} | Teacher: {cfg['teacher'] if args.mode != 'ce' else 'None (CE only)'}")
    print(f"Epochs:      {epochs} | Batch size: {batch_size} | Base LR: {base_lr}")
    print(f"Hyperparams: alpha={alpha}, beta={beta}, T={temperature}, warmup={warmup_epochs}")
    print(f"Output dir:  {output_dir}")

    train_loader, test_loader, _ = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=batch_size,
        eval_batch_size=128,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        seed=args.seed,
    )

    student = build_model(cfg["student"], num_classes=100).to(device)
    teacher = None
    if args.mode != "ce":
        teacher = load_pretrained_teacher(
            checkpoint_path=args.teacher_ckpt,
            num_classes=100,
            device=device,
            download_if_missing=True,
        )
        teacher_eval = evaluate(teacher, test_loader, device=device, max_batches=args.max_batches)
        print(
            f"Verified Teacher ({cfg['teacher']}) on test split: "
            f"top1={teacher_eval['top1']:.2f}%, top5={teacher_eval['top5']:.2f}%, "
            f"loss={teacher_eval['loss']:.4f}"
        )

    if args.benchmark_timing:
        bench_teacher = teacher if teacher is not None else build_model(cfg["teacher"], num_classes=100).to(device)
        timings = benchmark_batch_latency(
            student=student,
            teacher=bench_teacher,
            batch_size=batch_size,
            device=device,
        )
        print("Batch step latency benchmark (ms/batch):")
        for m_name, ms in timings.items():
            print(f"  {m_name:10s}: {ms:.2f} ms/batch")

    criterion = DistillationLoss(
        mode=args.mode,  # type: ignore[arg-type]
        ce_weight=ce_weight,
        alpha=alpha,
        beta=beta,
        temperature=temperature,
        warmup_epochs=warmup_epochs,
    )
    optimizer = build_optimizer(
        student,
        lr=base_lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    best_top1 = 0.0
    start_epoch = 1
    last_ckpt = output_dir / "ckpt_last.pth"
    if args.resume and last_ckpt.exists():
        start_epoch, best_top1 = load_resume_checkpoint(
            last_ckpt,
            student=student,
            optimizer=optimizer,
            map_location=device,
        )
        print(f"Resumed from {last_ckpt} -> starting at epoch {start_epoch} (best_top1={best_top1:.2f}%)")

    for epoch in range(start_epoch, epochs + 1):
        t_epoch = time.perf_counter()
        current_lr = adjust_learning_rate(
            optimizer,
            epoch=epoch,
            base_lr=base_lr,
            decay_stages=lr_decay_stages,
            decay_rate=lr_decay_rate,
        )

        train_stats = train_one_epoch(
            student=student,
            teacher=teacher,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            epoch=epoch,
            max_batches=args.max_batches,
            log_interval=args.log_interval,
        )
        test_stats = evaluate(
            model=student,
            data_loader=test_loader,
            device=device,
            max_batches=args.max_batches,
        )
        epoch_sec = time.perf_counter() - t_epoch

        is_best = test_stats["top1"] >= best_top1
        if is_best:
            best_top1 = test_stats["top1"]

        metrics = EpochMetrics(
            epoch=epoch,
            lr=current_lr,
            train_loss=train_stats["loss"],
            train_loss_ce=train_stats["loss_ce"],
            train_loss_distill=train_stats["loss_distill"],
            train_top1=train_stats["top1"],
            test_loss=test_stats["loss"],
            test_top1=test_stats["top1"],
            test_top5=test_stats["top5"],
            avg_batch_ms=train_stats["avg_batch_ms"],
            epoch_sec=epoch_sec,
        )
        append_metrics_jsonl(output_dir, metrics)

        save_checkpoint(
            output_dir=output_dir,
            filename="ckpt_last.pth",
            student=student,
            optimizer=optimizer,
            epoch=epoch,
            best_top1=best_top1,
            mode=args.mode,
            config=cfg,
        )
        if is_best:
            save_checkpoint(
                output_dir=output_dir,
                filename="ckpt_best.pth",
                student=student,
                optimizer=optimizer,
                epoch=epoch,
                best_top1=best_top1,
                mode=args.mode,
                config=cfg,
            )

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | lr={current_lr:.5f} | "
            f"train_loss={metrics.train_loss:.4f} | train_top1={metrics.train_top1:.2f}% | "
            f"test_top1={metrics.test_top1:.2f}% (best={best_top1:.2f}%) | "
            f"batch={metrics.avg_batch_ms:.1f}ms | {epoch_sec:.1f}s"
        )

    print(f"Training complete. Best test top-1 accuracy: {best_top1:.2f}% (saved in {output_dir})")


if __name__ == "__main__":
    main()
