"""Parameter-freezing policy for the staged audit-head training plan."""

from __future__ import annotations

from torch import nn

from audit_engine.models.heads import AuditHeads

STAGE_MODULES = {
    "B1": ("shared", "relation"),
    "B2": ("contribution",),
    "B3": ("shared", "relation", "contribution"),
    "B4": ("shared", "relation", "contribution"),
    "C": ("corrector",),
    "C2": ("shared", "relation", "contribution", "corrector"),
}


def _set_trainable(module: nn.Module, is_trainable: bool) -> None:
    module.train(is_trainable)
    for parameter in module.parameters():
        parameter.requires_grad_(is_trainable)
        parameter.grad = None


def configure_trainable_stage(model: AuditHeads, stage: str) -> None:
    """Apply the guide's trainable/frozen module policy for one stage."""
    if stage not in STAGE_MODULES:
        raise ValueError(f"unknown training stage: {stage!r}")
    trainable = set(STAGE_MODULES[stage])
    for name in ("shared", "relation", "contribution", "corrector"):
        _set_trainable(getattr(model, name), name in trainable)
