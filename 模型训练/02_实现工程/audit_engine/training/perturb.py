"""Construct bounded internal perturbations and retain detected genuine errors."""

from __future__ import annotations

import torch
from torch import Tensor

from audit_engine.models.heads import AuditHeads
from audit_engine.training.forward import (
    ScalarCondition,
    correction_inputs,
    forward_condition,
)
from audit_engine.training.losses import masked_reference_loss, masked_relation_loss


def bounded_perturbation(
    condition: Tensor, gradient: Tensor, rho_relative: float = 0.03
) -> Tensor:
    if not 0 < rho_relative <= 1:
        raise ValueError("invalid relative perturbation radius")
    radius = rho_relative * condition.detach().norm(dim=-1, keepdim=True)
    delta = radius * gradient / (gradient.norm(dim=-1, keepdim=True) + 1e-8)
    return condition.detach() + delta.detach()


def generate_detectable_errors(
    model: AuditHeads,
    condition: Tensor,
    evidence: Tensor,
    policies: list[ScalarCondition],
    active: Tensor,
    labels: dict[str, Tensor],
    rho_relative: float = 0.03,
    thresholds: dict[str, float] | None = None,
    reference_verified: Tensor | None = None,
) -> tuple[Tensor, Tensor, dict]:
    """Gold only selects perturbations; runtime detection never receives labels."""
    limits = thresholds or {"reference": 0.6, "support": 0.7, "conflict": 0.7}
    if "state" not in labels or "state_mask" not in labels:
        raise ValueError("perturbation requires an approved condition-state target")
    if (
        labels["state"].shape != condition.shape[:1]
        or labels["state_mask"].shape != condition.shape[:1]
    ):
        raise ValueError(
            "condition-state target or approval mask has incompatible shape"
        )
    h = condition.detach().clone().requires_grad_(True)
    before = forward_condition(model, h, evidence.detach(), policies, active)
    loss = masked_relation_loss(
        before["logits"], labels["relation"], labels["relation_mask"]
    )
    reference = masked_reference_loss(
        before["references"], labels["reference"], labels["reference_mask"]
    )
    if loss is None or reference is None:
        raise ValueError("perturbation requires verified relation and reference labels")
    (gradient,) = torch.autograd.grad(loss + reference, h)
    perturbed = bounded_perturbation(h, gradient, rho_relative)
    with torch.no_grad():
        after = forward_condition(model, perturbed, evidence.detach(), policies, active)
        _, after_fact, _, eligible = correction_inputs(
            evidence.detach(), after, active, policies, limits,
            reference_verified=reference_verified,
        )
        _, before_fact, _, _ = correction_inputs(
            evidence.detach(), before, active, policies, limits,
            reference_verified=reference_verified,
        )
        state_before = before_fact[..., -4:].argmax(-1) == labels["state"]
        state_after = after_fact[..., -4:].argmax(-1) == labels["state"]
        relation_before = (
            (before["logits"].argmax(-1) == labels["relation"])
            | ~labels["relation_mask"]
        ).all(-1)
        reference_before = (
            (
                (before["references"].sigmoid() >= limits["reference"])
                == labels["reference"].bool()
            )
            | ~labels["reference_mask"]
        ).all(dim=(-1, -2))
        relation_after = (
            (after["logits"].argmax(-1) == labels["relation"])
            | ~labels["relation_mask"]
        ).all(-1)
        reference_after = (
            (
                (after["references"].sigmoid() >= limits["reference"])
                == labels["reference"].bool()
            )
            | ~labels["reference_mask"]
        ).all(dim=(-1, -2))
        original_correct = relation_before & reference_before & state_before
        original_correct &= labels["state_mask"].bool()
        created_error = ~relation_after | ~reference_after | ~state_after
        accepted = original_correct & created_error & eligible
    return (
        perturbed,
        accepted,
        {
            "candidate_count": condition.size(0),
            "original_correct_count": int(original_correct.sum()),
            "created_error_count": int((original_correct & created_error).sum()),
            "accepted_detectable_count": int(accepted.sum()),
            "rho_relative": rho_relative,
            "max_relative_norm": float(
                (
                    (perturbed - condition).norm(dim=-1)
                    / condition.norm(dim=-1).clamp_min(1e-8)
                ).max()
            ),
            "original_facts_changed": False,
        },
    )
