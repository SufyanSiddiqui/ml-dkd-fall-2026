"""Evaluate a CIFAR-100 ResNet checkpoint (teacher or student) and optionally benchmark batch timing.

Examples:
    # Verify the official pretrained ResNet32x4 teacher (~79.42% top-1):
    .venv/bin/python scripts/eval.py --arch resnet32x4 --download-teacher

    # Evaluate a trained ResNet8x4 student checkpoint:
    .venv/bin/python scripts/eval.py --arch resnet8x4 --ckpt outputs/dkd/ckpt_best.pth

    # Run the Category-A batch-latency comparison (CE vs KD vs DKD vs TCKD-only vs NCKD-only):
    .venv/bin/python scripts/eval.py --benchmark-timing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.cifar100 import DEFAULT_DATA_DIR, build_dataloaders
from src.engine.trainer import benchmark_batch_latency, evaluate, select_device
from src.models.resnet import (
    MODEL_REGISTRY,
    build_model,
    load_checkpoint_state_dict,
    load_pretrained_teacher,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CIFAR-100 model checkpoints.")
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet32x4",
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Model architecture to evaluate.",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Path to checkpoint file.",
    )
    parser.add_argument(
        "--download-teacher",
        action="store_true",
        help="Use (and download if missing) the official ResNet32x4 vanilla teacher checkpoint.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to extracted CIFAR-100 dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto | cuda | mps | cpu.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional max batches for smoke evaluation.",
    )
    parser.add_argument(
        "--benchmark-timing",
        action="store_true",
        help="Benchmark ms/batch training latency across all 5 loss modes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    if args.benchmark_timing:
        student = build_model("resnet8x4", num_classes=100).to(device)
        teacher = build_model("resnet32x4", num_classes=100).to(device)
        timings = benchmark_batch_latency(
            student=student,
            teacher=teacher,
            batch_size=64,
            device=device,
        )
        print(f"Batch step latency on {device} (batch_size=64):")
        for mode_name, ms in timings.items():
            print(f"  {mode_name:10s}: {ms:.2f} ms/batch")
        if not args.download_teacher and args.ckpt is None:
            return

    _, test_loader, _ = build_dataloaders(
        data_dir=args.data_dir,
        eval_batch_size=args.batch_size,
    )

    if args.download_teacher:
        model = load_pretrained_teacher(
            checkpoint_path=args.ckpt,
            num_classes=100,
            device=device,
            download_if_missing=True,
        )
    else:
        if args.ckpt is None:
            raise SystemExit("Pass --ckpt <path> or --download-teacher to evaluate a checkpoint.")
        model = build_model(args.arch, num_classes=100)
        state = load_checkpoint_state_dict(args.ckpt, map_location=device)
        model.load_state_dict(state, strict=True)
        model.to(device)

    stats = evaluate(model, test_loader, device=device, max_batches=args.max_batches)
    print(
        f"[{args.arch}] samples={int(stats['samples'])} | "
        f"top-1={stats['top1']:.2f}% | top-5={stats['top5']:.2f}% | "
        f"ce_loss={stats['loss']:.4f}"
    )


if __name__ == "__main__":
    main()
