"""From-scratch PyTorch implementation of Classical KD and Decoupled KD (DKD).

Reference paper:
    Borui Zhao, Quan Cui, Renjie Song, Yiyu Qiu, and Jiajun Liang,
    "Decoupled Knowledge Distillation," CVPR 2022, pp. 11953-11962.

Key equations:
    1. Binary target vs. non-target distribution for true class t:
       b = [p_t, p_{\\t}], where p_t = exp(z_t / T) / sum_j exp(z_j / T)
       and p_{\\t} = 1 - p_t.

    2. Non-target conditional distribution over i != t:
       \\hat{p}_i = exp(z_i / T) / sum_{j != t} exp(z_j / T).

    3. Exact reformulation of classical KD (Eq. 1):
       KL(p^T || p^S) = KL(b^T || b^S) + (1 - p_t^T) * KL(\\hat{p}^T || \\hat{p}^S)
                      = TCKD           + (1 - p_t^T) * NCKD

    4. Decoupled Knowledge Distillation (Eq. 2):
       L_DKD = (alpha * TCKD + beta * NCKD) * T^2
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

LossMode = Literal["ce", "kd", "dkd", "tckd_only", "nckd_only"]

_LOGIT_MASK_VALUE = 1e3


def _get_gt_mask(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Boolean mask of shape (B, C) with True at the ground-truth class index."""
    target_idx = target.reshape(-1, 1)
    return torch.zeros_like(logits, dtype=torch.bool).scatter_(1, target_idx, True)


def _get_other_mask(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Boolean mask of shape (B, C) with True at all non-target class indices."""
    target_idx = target.reshape(-1, 1)
    return torch.ones_like(logits, dtype=torch.bool).scatter_(1, target_idx, False)


def _cat_mask(
    probs: torch.Tensor,
    gt_mask: torch.Tensor,
    other_mask: torch.Tensor,
) -> torch.Tensor:
    """Construct binary target/non-target probability vector b = [p_t, 1 - p_t] of shape (B, 2)."""
    p_t = (probs * gt_mask).sum(dim=1, keepdim=True)
    p_not_t = (probs * other_mask).sum(dim=1, keepdim=True)
    return torch.cat([p_t, p_not_t], dim=1)


def tckd_per_sample(
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 4.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute per-sample TCKD = KL(b^T || b^S) without T^2 scaling. Shape: (B,)."""
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)

    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)

    b_student = _cat_mask(pred_student, gt_mask, other_mask).clamp(min=eps)
    b_teacher = _cat_mask(pred_teacher, gt_mask, other_mask).clamp(min=eps)

    # KL(b^T || b^S) = sum_{k in {t, \t}} b_k^T * (log b_k^T - log b_k^S)
    return F.kl_div(b_student.log(), b_teacher, reduction="none").sum(dim=1)


def nckd_per_sample(
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """Compute per-sample NCKD = KL(\\hat{p}^T || \\hat{p}^S) without T^2 scaling. Shape: (B,)."""
    gt_mask = _get_gt_mask(logits_student, target)

    # Subtract a large constant on the target class before softmax so exp(z_t/T - 1000) == 0,
    # matching mdistiller/distillers/DKD.py while keeping tensors rectangular.
    masked_student = logits_student / temperature - _LOGIT_MASK_VALUE * gt_mask.to(logits_student.dtype)
    masked_teacher = logits_teacher / temperature - _LOGIT_MASK_VALUE * gt_mask.to(logits_teacher.dtype)

    log_pred_student_nckd = F.log_softmax(masked_student, dim=1)
    pred_teacher_nckd = F.softmax(masked_teacher, dim=1)

    return F.kl_div(log_pred_student_nckd, pred_teacher_nckd, reduction="none").sum(dim=1)


def tckd_loss(
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """Batch-averaged TCKD scaled by T^2."""
    return tckd_per_sample(logits_student, logits_teacher, target, temperature).mean() * (
        temperature**2
    )


def nckd_loss(
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """Batch-averaged NCKD scaled by T^2."""
    return nckd_per_sample(logits_student, logits_teacher, target, temperature).mean() * (
        temperature**2
    )


def dkd_loss(
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 8.0,
    temperature: float = 4.0,
) -> torch.Tensor:
    """Decoupled Knowledge Distillation loss: (alpha * TCKD + beta * NCKD) * T^2."""
    tckd = tckd_per_sample(logits_student, logits_teacher, target, temperature).mean()
    nckd = nckd_per_sample(logits_student, logits_teacher, target, temperature).mean()
    return (alpha * tckd + beta * nckd) * (temperature**2)


def kd_loss(
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """Classical Hinton Knowledge Distillation loss: KL(p^T || p^S) * T^2."""
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    return F.kl_div(log_pred_student, pred_teacher, reduction="batchmean") * (temperature**2)


def kd_loss_from_decoupled_terms(
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """Compute classical KD via Eq. (1): mean(TCKD_i + (1 - p_{t,i}^T) * NCKD_i) * T^2.

    Used to verify the exact algebraic identity in Zhao et al. (CVPR 2022).
    """
    gt_mask = _get_gt_mask(logits_teacher, target)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pt_teacher = (pred_teacher * gt_mask).sum(dim=1)  # shape (B,)

    tckd_i = tckd_per_sample(logits_student, logits_teacher, target, temperature)
    nckd_i = nckd_per_sample(logits_student, logits_teacher, target, temperature)
    coupled = tckd_i + (1.0 - pt_teacher) * nckd_i
    return coupled.mean() * (temperature**2)


def warmup_factor(epoch: int, warmup_epochs: int = 20) -> float:
    """Linear warmup multiplier min(epoch / warmup_epochs, 1.0) matching mdistiller DKD.

    Note: epoch is 1-indexed in mdistiller (epoch=1 -> 1/20, ..., epoch>=20 -> 1.0).
    If warmup_epochs <= 0, returns 1.0.
    """
    if warmup_epochs <= 0:
        return 1.0
    if epoch <= 0:
        return 0.0
    return float(min(epoch / warmup_epochs, 1.0))


class DistillationLoss(nn.Module):
    """Unified loss wrapper for reproduction and Category-A ablation runs.

    Supported modes:
        - 'ce':        hard-label cross-entropy only (ResNet8x4 baseline)
        - 'kd':        CE + kd_weight * classical KD
        - 'dkd':       CE + warmup * DKD(alpha=alpha, beta=beta)
        - 'tckd_only': CE + warmup * DKD(alpha=alpha, beta=0.0)
        - 'nckd_only': CE + warmup * DKD(alpha=0.0, beta=beta)
    """

    VALID_MODES: tuple[str, ...] = ("ce", "kd", "dkd", "tckd_only", "nckd_only")

    def __init__(
        self,
        mode: LossMode = "dkd",
        ce_weight: float = 1.0,
        alpha: float = 1.0,
        beta: float = 8.0,
        temperature: float = 4.0,
        warmup_epochs: int = 20,
        kd_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode {mode!r}. Expected one of {self.VALID_MODES}")
        self.mode: LossMode = mode
        self.ce_weight = float(ce_weight)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.temperature = float(temperature)
        self.warmup_epochs = int(warmup_epochs)
        self.kd_weight = float(kd_weight)

    def forward(
        self,
        logits_student: torch.Tensor,
        target: torch.Tensor,
        logits_teacher: torch.Tensor | None = None,
        epoch: int = 20,
    ) -> dict[str, torch.Tensor]:
        """Compute total loss and breakdown dict {'loss', 'loss_ce', 'loss_distill'}."""
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)

        if self.mode == "ce":
            zero = torch.zeros_like(loss_ce)
            return {
                "loss": loss_ce,
                "loss_ce": loss_ce,
                "loss_distill": zero,
            }

        if logits_teacher is None:
            raise ValueError(f"logits_teacher is required when mode={self.mode!r}")

        if self.mode == "kd":
            loss_distill = self.kd_weight * kd_loss(
                logits_student,
                logits_teacher,
                temperature=self.temperature,
            )
        else:
            eff_alpha = 0.0 if self.mode == "nckd_only" else self.alpha
            eff_beta = 0.0 if self.mode == "tckd_only" else self.beta
            w = warmup_factor(epoch=epoch, warmup_epochs=self.warmup_epochs)
            loss_distill = w * dkd_loss(
                logits_student,
                logits_teacher,
                target,
                alpha=eff_alpha,
                beta=eff_beta,
                temperature=self.temperature,
            )

        total = loss_ce + loss_distill
        return {
            "loss": total,
            "loss_ce": loss_ce,
            "loss_distill": loss_distill,
        }
