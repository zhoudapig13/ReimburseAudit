"""Masked supervision losses specified for audit-head training."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as functional


def _condition_average(losses: Tensor, approved: Tensor) -> Tensor | None:
    flat_losses = losses.flatten(start_dim=1)
    flat_approved = approved.flatten(start_dim=1)
    counts = flat_approved.sum(dim=1)
    supervised = counts > 0
    if not bool(supervised.any()):
        return None
    sums = (flat_losses * flat_approved.to(losses.dtype)).sum(dim=1)
    per_condition = sums[supervised] / counts[supervised].to(losses.dtype)
    return per_condition.mean()


def masked_relation_loss(
    logits: Tensor, targets: Tensor, approved: Tensor,
    class_weights: Tensor | None = None,
) -> Tensor | None:
    """Average approved relation labels per condition and then across batch."""
    if logits.shape[:-1] != targets.shape or logits.size(-1) != 3:
        raise ValueError("relation logits and targets have incompatible shapes")
    if approved.shape != targets.shape:
        raise ValueError("relation approval mask has an incompatible shape")
    if class_weights is not None:
        if class_weights.shape != (3,) or not bool(torch.isfinite(class_weights).all()) or bool((class_weights <= 0).any()):
            raise ValueError("relation class weights must contain three finite positive values")
        class_weights = class_weights.to(device=logits.device, dtype=logits.dtype)
    valid = approved.bool() & (targets != -100)
    safe_targets = targets.masked_fill(~valid, 0)
    losses = functional.cross_entropy(
        logits.reshape(-1, 3), safe_targets.reshape(-1), weight=class_weights, reduction="none"
    ).reshape_as(targets)
    return _condition_average(losses, valid)


def masked_reference_loss(
    logits: Tensor, targets: Tensor, approved: Tensor
) -> Tensor | None:
    """Average approved support/conflict reference labels per condition."""
    if logits.shape != targets.shape or logits.shape != approved.shape:
        raise ValueError("reference tensors must have identical shapes")
    losses = functional.binary_cross_entropy_with_logits(
        logits, targets.to(logits.dtype), reduction="none"
    )
    return _condition_average(losses, approved.bool())


def state_probabilities(strengths: Tensor) -> Tensor:
    """Map support/conflict strengths to satisfied/failed/gap/conflict order."""
    if strengths.ndim != 2 or strengths.size(-1) != 2:
        raise ValueError("strengths must have shape (batch, 2)")
    if bool(((strengths < 0) | (strengths > 1)).any()):
        raise ValueError("strengths must be within [0, 1]")
    support, conflict = strengths.unbind(dim=-1)
    return torch.stack(
        [
            support * (1 - conflict),
            (1 - support) * conflict,
            (1 - support) * (1 - conflict),
            support * conflict,
        ],
        dim=-1,
    )


def masked_state_loss(
    probabilities: Tensor, targets: Tensor, approved: Tensor
) -> Tensor | None:
    """Return negative log likelihood for approved condition-level states."""
    if probabilities.ndim != 2 or probabilities.size(-1) != 4:
        raise ValueError("state probabilities must have shape (batch, 4)")
    if targets.shape != probabilities.shape[:1] or approved.shape != targets.shape:
        raise ValueError("state targets or approval mask have an incompatible shape")
    valid = approved.bool() & (targets != -100)
    if not bool(valid.any()):
        return None
    safe_targets = targets.masked_fill(~valid, 0)
    selected = probabilities.gather(1, safe_targets[:, None]).squeeze(1)
    return -torch.log(selected[valid].clamp(1e-6, 1)).mean()
