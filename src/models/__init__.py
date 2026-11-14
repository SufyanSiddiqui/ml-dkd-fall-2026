"""Model definitions for CIFAR-100 knowledge distillation."""

from src.models.resnet import (
    BasicBlock,
    CifarResNet,
    build_model,
    download_teacher_checkpoint,
    load_checkpoint_state_dict,
    load_pretrained_teacher,
    resnet8x4,
    resnet32x4,
)

__all__ = [
    "BasicBlock",
    "CifarResNet",
    "build_model",
    "download_teacher_checkpoint",
    "load_checkpoint_state_dict",
    "load_pretrained_teacher",
    "resnet8x4",
    "resnet32x4",
]
