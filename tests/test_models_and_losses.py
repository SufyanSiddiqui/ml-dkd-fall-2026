"""Unit tests for CIFAR ResNet models, DKD/KD losses, and CIFAR-100 dataset wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.cifar100 import CIFAR100Dataset, preprocess_pil_eval
from src.losses.dkd import (
    DistillationLoss,
    dkd_loss,
    kd_loss,
    kd_loss_from_decoupled_terms,
    nckd_loss,
    nckd_per_sample,
    tckd_loss,
    tckd_per_sample,
    warmup_factor,
)
from src.models.resnet import (
    build_model,
    load_checkpoint_state_dict,
    load_pretrained_teacher,
    resnet8x4,
    resnet32x4,
)


def test_worked_numeric_identity_three_classes() -> None:
    """Verify the exact 3-class worked example: teacher [4, 1, 0], student [3, 1.5, 0], T=1."""
    logits_t = torch.tensor([[4.0, 1.0, 0.0]], dtype=torch.float64)
    logits_s = torch.tensor([[3.0, 1.5, 0.0]], dtype=torch.float64)
    target = torch.tensor([0], dtype=torch.long)

    tckd = tckd_per_sample(logits_s, logits_t, target, temperature=1.0).item()
    nckd = nckd_per_sample(logits_s, logits_t, target, temperature=1.0).item()
    kd_classical = kd_loss(logits_s, logits_t, temperature=1.0).item()
    kd_reconstructed = kd_loss_from_decoupled_terms(logits_s, logits_t, target, temperature=1.0).item()

    assert tckd == pytest.approx(0.086918, abs=1e-5)
    assert nckd == pytest.approx(0.022622, abs=1e-5)
    assert kd_classical == pytest.approx(0.088361, abs=1e-5)
    assert kd_reconstructed == pytest.approx(kd_classical, abs=1e-10)


def test_exact_algebraic_identity_random_batches() -> None:
    """Verify KL(p^T || p^S) == TCKD + (1 - p_t^T) * NCKD across random batches and temperatures."""
    torch.manual_seed(42)
    batch_size, num_classes = 32, 100
    logits_s = torch.randn(batch_size, num_classes, dtype=torch.float64)
    logits_t = torch.randn(batch_size, num_classes, dtype=torch.float64) * 2.0
    target = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)

    for temp in (1.0, 2.0, 4.0):
        kd_std = kd_loss(logits_s, logits_t, temperature=temp)
        kd_decoupled = kd_loss_from_decoupled_terms(logits_s, logits_t, target, temperature=temp)
        assert torch.allclose(kd_std, kd_decoupled, atol=1e-7), (
            f"Mismatch at T={temp}: {kd_std.item()} vs {kd_decoupled.item()}"
        )


def test_dkd_ablation_modes_and_gradients() -> None:
    """Check CE, KD, DKD, TCKD-only, and NCKD-only loss modes and gradient flow to student."""
    torch.manual_seed(7)
    logits_s = torch.randn(8, 100, requires_grad=True)
    logits_t = torch.randn(8, 100)
    target = torch.randint(0, 100, (8,))

    t_val = tckd_loss(logits_s, logits_t, target, temperature=4.0)
    n_val = nckd_loss(logits_s, logits_t, target, temperature=4.0)
    d_val = dkd_loss(logits_s, logits_t, target, alpha=1.0, beta=8.0, temperature=4.0)

    assert torch.allclose(d_val, 1.0 * t_val + 8.0 * n_val, atol=1e-6)

    # TCKD-only mode (alpha=1, beta=0) at full warmup (epoch=20)
    loss_tckd_only = DistillationLoss(mode="tckd_only", alpha=1.0, beta=8.0, temperature=4.0, warmup_epochs=20)
    out_tckd = loss_tckd_only(logits_s, target, logits_t, epoch=20)
    assert torch.allclose(out_tckd["loss_distill"], t_val, atol=1e-6)

    # NCKD-only mode (alpha=0, beta=8) at full warmup (epoch=20)
    loss_nckd_only = DistillationLoss(mode="nckd_only", alpha=1.0, beta=8.0, temperature=4.0, warmup_epochs=20)
    out_nckd = loss_nckd_only(logits_s, target, logits_t, epoch=20)
    assert torch.allclose(out_nckd["loss_distill"], 8.0 * n_val, atol=1e-6)

    # Warmup scaling at epoch 5 / 20 -> 0.25
    loss_dkd = DistillationLoss(mode="dkd", alpha=1.0, beta=8.0, temperature=4.0, warmup_epochs=20)
    out_warmup = loss_dkd(logits_s, target, logits_t, epoch=5)
    assert torch.allclose(out_warmup["loss_distill"], 0.25 * d_val, atol=1e-6)

    out_warmup["loss"].backward()
    assert logits_s.grad is not None
    assert torch.isfinite(logits_s.grad).all()


def test_warmup_factor_schedule() -> None:
    assert warmup_factor(0, 20) == 0.0
    assert warmup_factor(1, 20) == pytest.approx(0.05)
    assert warmup_factor(10, 20) == pytest.approx(0.50)
    assert warmup_factor(20, 20) == pytest.approx(1.0)
    assert warmup_factor(150, 20) == pytest.approx(1.0)
    assert warmup_factor(1, 0) == pytest.approx(1.0)


def test_resnet_architectures_and_checkpoint_loading(tmp_path: Path) -> None:
    """Verify ResNet8x4 and ResNet32x4 shapes, layer counts, and checkpoint round-trip."""
    student = build_model("resnet8x4", num_classes=100)
    teacher = build_model("resnet32x4", num_classes=100)

    assert len(student.layer1) == 1
    assert len(student.layer2) == 1
    assert len(student.layer3) == 1

    assert len(teacher.layer1) == 5
    assert len(teacher.layer2) == 5
    assert len(teacher.layer3) == 5

    student_params = sum(p.numel() for p in student.parameters())
    teacher_params = sum(p.numel() for p in teacher.parameters())
    assert student_params < teacher_params

    x = torch.randn(4, 3, 32, 32)
    out_s = student(x)
    out_t = teacher(x)
    assert out_s.shape == (4, 100)
    assert out_t.shape == (4, 100)

    # Simulate mdistiller teacher checkpoint format {'model': state_dict}
    ckpt_file = tmp_path / "teacher_ckpt.pth"
    wrapped_state = {"model": {f"module.{k}": v for k, v in teacher.state_dict().items()}}
    torch.save(wrapped_state, ckpt_file)

    cleaned = load_checkpoint_state_dict(ckpt_file)
    assert "conv1.weight" in cleaned
    loaded_teacher = load_pretrained_teacher(checkpoint_path=ckpt_file, device="cpu")
    assert not loaded_teacher.training
    assert all(not p.requires_grad for p in loaded_teacher.parameters())


def test_cifar100_dataset_wrapper_and_pil_preprocess() -> None:
    """Verify CIFAR100Dataset and PIL eval helper return float32 CHW (3, 32, 32) tensors."""
    rng = np.random.default_rng(0)
    fake_images = rng.integers(0, 256, size=(10, 32, 32, 3), dtype=np.uint8)
    fake_labels = np.arange(10, dtype=np.int64)

    train_ds = CIFAR100Dataset(fake_images, fake_labels, train=True, seed=123)
    eval_ds = CIFAR100Dataset(fake_images, fake_labels, train=False)

    assert len(train_ds) == 10
    img_t, lbl_t = train_ds[0]
    img_e, lbl_e = eval_ds[0]

    assert img_t.shape == (3, 32, 32)
    assert img_t.dtype == torch.float32
    assert lbl_t == 0

    assert img_e.shape == (3, 32, 32)
    assert img_e.dtype == torch.float32
    assert lbl_e == 0

    pil_img = Image.fromarray(fake_images[0])
    chw = preprocess_pil_eval(pil_img)
    assert chw.shape == (3, 32, 32)
    assert np.allclose(chw, img_e.numpy(), atol=1e-6)
