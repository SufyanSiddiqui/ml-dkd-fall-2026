"""CIFAR ResNet architectures (ResNet32x4 teacher and ResNet8x4 student).

Follows the standard CIFAR ResNet formulation (He et al., 2016) with the 4x
channel width multiplier [32, 64, 128, 256] used in Decoupled Knowledge
Distillation (Zhao et al., CVPR 2022). Parameter names and shapes match
mdistiller's CIFAR ResNet so official plain-CE teacher checkpoints can be
loaded with strict=True.
"""

from __future__ import annotations

import math
import urllib.request
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_DIR = _ROOT / "checkpoints"
RESNET32X4_TEACHER_URL = (
    "https://github.com/megvii-research/mdistiller/releases/download/checkpoints/cifar_teachers.tar"
)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding=1 and no bias."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    """Standard 2-layer residual BasicBlock for CIFAR ResNets."""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)
        return out


class CifarResNet(nn.Module):
    """3-stage CIFAR ResNet for 32x32 RGB inputs.

    Depth must satisfy (depth - 2) % 6 == 0 for BasicBlock.
    - ResNet8x4:  depth=8  -> n=1 block per stage, filters [32, 64, 128, 256]
    - ResNet32x4: depth=32 -> n=5 blocks per stage, filters [32, 64, 128, 256]
    """

    def __init__(
        self,
        depth: int,
        num_filters: list[int],
        num_classes: int = 100,
    ) -> None:
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError(f"CIFAR ResNet BasicBlock depth must satisfy 6n + 2, got {depth}.")
        n = (depth - 2) // 6

        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock, num_filters[1], n, stride=1)
        self.layer2 = self._make_layer(BasicBlock, num_filters[2], n, stride=2)
        self.layer3 = self._make_layer(BasicBlock, num_filters[3], n, stride=2)

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(num_filters[3] * BasicBlock.expansion, num_classes)

        self._init_weights()

    def _make_layer(
        self,
        block: type[BasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers: list[nn.Module] = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)  # 1x1
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits


def resnet8x4(num_classes: int = 100) -> CifarResNet:
    """Construct the CIFAR ResNet8x4 student network."""
    return CifarResNet(depth=8, num_filters=[32, 64, 128, 256], num_classes=num_classes)


def resnet32x4(num_classes: int = 100) -> CifarResNet:
    """Construct the CIFAR ResNet32x4 teacher network."""
    return CifarResNet(depth=32, num_filters=[32, 64, 128, 256], num_classes=num_classes)


MODEL_REGISTRY: dict[str, Callable[..., CifarResNet]] = {
    "resnet8x4": resnet8x4,
    "resnet32x4": resnet32x4,
}


def build_model(name: str, num_classes: int = 100) -> CifarResNet:
    """Instantiate a CIFAR model by name."""
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key](num_classes=num_classes)


def load_checkpoint_state_dict(
    checkpoint_path: Path | str,
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load a state_dict from a local checkpoint file.

    Handles both plain state_dicts and wrapped dicts (e.g. mdistiller's
    {'model': state_dict, ...} or {'state_dict': state_dict, ...}) and strips
    an optional 'module.' DataParallel prefix.
    """
    raw = torch.load(Path(checkpoint_path), map_location=map_location, weights_only=False)
    if isinstance(raw, dict):
        if "model" in raw and isinstance(raw["model"], dict):
            state_dict = raw["model"]
        elif "state_dict" in raw and isinstance(raw["state_dict"], dict):
            state_dict = raw["state_dict"]
        else:
            state_dict = raw
    else:
        raise TypeError(f"Expected checkpoint dict, got {type(raw).__name__}")

    cleaned: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        new_key = k[len("module.") :] if k.startswith("module.") else k
        cleaned[new_key] = v
    return cleaned


def load_pretrained_teacher(
    checkpoint_path: Path | str | None = None,
    num_classes: int = 100,
    device: str | torch.device = "cpu",
    download_if_missing: bool = False,
) -> CifarResNet:
    """Build ResNet32x4 in eval mode (frozen) and optionally load weights.

    If checkpoint_path is None, looks in checkpoints/resnet32x4_vanilla/ckpt_epoch_240.pth.
    If download_if_missing is True and the checkpoint is not on disk, downloads
    and extracts the official mdistiller cifar_teachers.tar archive.
    """
    teacher = resnet32x4(num_classes=num_classes)
    if checkpoint_path is None:
        default_ckpt = DEFAULT_CHECKPOINT_DIR / "cifar_teachers" / "resnet32x4_vanilla" / "ckpt_epoch_240.pth"
        if default_ckpt.exists() or download_if_missing:
            checkpoint_path = default_ckpt

    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists() and download_if_missing:
            download_teacher_checkpoint(DEFAULT_CHECKPOINT_DIR)
        state_dict = load_checkpoint_state_dict(ckpt_path, map_location=device)
        teacher.load_state_dict(state_dict, strict=True)

    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def download_teacher_checkpoint(dest_dir: Path | str = DEFAULT_CHECKPOINT_DIR) -> Path:
    """Download and extract the official mdistiller resnet32x4_vanilla checkpoint."""
    import ssl
    import tarfile

    dest = Path(dest_dir)
    ckpt_path = dest / "cifar_teachers" / "resnet32x4_vanilla" / "ckpt_epoch_240.pth"
    if ckpt_path.exists():
        return ckpt_path

    dest.mkdir(parents=True, exist_ok=True)
    tar_path = dest / "cifar_teachers.tar"
    if not tar_path.exists():
        try:
            import certifi

            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ssl_ctx = ssl.create_default_context()

        request = urllib.request.Request(
            RESNET32X4_TEACHER_URL,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with (
            urllib.request.urlopen(request, timeout=120, context=ssl_ctx) as response,
            tar_path.open("wb") as out,
        ):
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)

    with tarfile.open(tar_path, "r") as archive:
        member_prefix = "cifar_teachers/resnet32x4_vanilla/"
        members = [m for m in archive.getmembers() if m.name.startswith(member_prefix)]
        if hasattr(tarfile, "data_filter"):
            archive.extractall(dest, members=members, filter="data")
        else:
            archive.extractall(dest, members=members)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Expected teacher checkpoint at {ckpt_path} after extraction.")
    if tar_path.exists():
        tar_path.unlink()
    return ckpt_path
