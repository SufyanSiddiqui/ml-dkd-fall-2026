"""Training, evaluation, and latency benchmarking engine."""

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
    select_device,
    set_seed,
    train_one_epoch,
)

__all__ = [
    "EpochMetrics",
    "adjust_learning_rate",
    "append_metrics_jsonl",
    "benchmark_batch_latency",
    "build_optimizer",
    "compute_accuracy",
    "evaluate",
    "load_config",
    "load_resume_checkpoint",
    "save_checkpoint",
    "select_device",
    "set_seed",
    "train_one_epoch",
]
