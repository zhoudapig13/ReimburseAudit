"""Explicit synthetic boundary diagnostic for gradient-based local recovery.

The engineering diagnostic uses a deliberately configured relation boundary.
This tests optimization on a detected bounded perturbation, never accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from audit_engine.models.heads import AuditHeads
from audit_engine.training.forward import (
    ScalarCondition,
    correction_inputs,
    forward_condition,
)
from audit_engine.training.losses import masked_relation_loss
from audit_engine.training.perturb import generate_detectable_errors
from audit_engine.training.stages import configure_trainable_stage


def run_perturbation_diagnostic(
    output_dir: Path, device: str, steps: int, config: dict
) -> dict:
    """Verify C gradients, loss decrease, freezing, and a missing-fact bypass."""
    if steps < 2:
        raise ValueError("diagnostic needs at least two optimization steps")
    # Preserve the caller's RNG stream; this diagnostic is independently seeded.
    cuda_devices = (
        [torch.device(device).index or 0] if device.startswith("cuda") else []
    )
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(config["project"]["seed"])
        model = AuditHeads().to(device)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not name.startswith("corrector."):
                parameter.zero_()
        model.relation.semantic[0].weight[0, 0] = 1
        model.relation.semantic[2].weight[0, 0] = 1
        model.relation.classifier.weight[0, 0] = 1
        model.relation.classifier.weight[1, 0] = -1
        model.relation.classifier.bias[2] = -4
        model.contribution.support.bias.fill_(8)
        model.contribution.conflict.bias.fill_(-8)
    configure_trainable_stage(model, "C")
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    condition = torch.zeros(2, 512, device=device)
    condition[:, 0], condition[:, 1] = 1.5, 100
    evidence = torch.zeros(2, 2, 512, device=device)
    original_condition, original_evidence = condition.clone(), evidence.clone()
    policies = [ScalarCondition("LE", (("90", None), (None, "100")), (0, 1))] * 2
    active = torch.tensor([[True, True], [True, False]], device=device)
    labels = {
        "relation": torch.tensor([[0, 0], [2, 2]], device=device),
        "relation_mask": active,
        "reference": torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]], device=device
        ),
        "reference_mask": active[..., None].expand(-1, -1, 2),
        "state": torch.tensor([0, 2], device=device),
        "state_mask": torch.ones(2, dtype=torch.bool, device=device),
    }
    perturbed, accepted, perturb_report = generate_detectable_errors(
        model,
        condition,
        evidence,
        policies,
        active,
        labels,
        config["correction"]["rho_relative"],
        config["thresholds"],
    )
    if accepted.tolist() != [True, False]:
        raise RuntimeError("controlled bounded-perturbation fixture failed")
    # A rejected candidate must not replace the original condition vector.
    initial = torch.where(accepted[:, None], perturbed, condition).detach()
    optimizer = torch.optim.AdamW(
        model.corrector.parameters(), lr=config["correction"]["learning_rate"]
    )
    losses, gradients = [], []
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "perturbation_diagnostic_metrics.jsonl").open("w") as log:
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            corrected = initial
            output = forward_condition(model, corrected, evidence, policies, active)
            for _ in range(config["correction"]["max_rounds"]):
                errors, fact, slots, eligible = correction_inputs(
                    evidence, output, active, policies, config["thresholds"]
                )
                corrected = model.corrector(
                    corrected, output["context"], errors, fact, slots, eligible
                )
                output = forward_condition(model, corrected, evidence, policies, active)
            if not torch.equal(corrected[~accepted], initial[~accepted]):
                raise RuntimeError("missing-fact condition changed")
            loss = masked_relation_loss(
                output["logits"], labels["relation"], active & accepted[:, None]
            )
            if loss is None or not loss.requires_grad or not torch.isfinite(loss):
                raise RuntimeError("invalid correction loss or absent gradient")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.corrector.parameters(), 1.0)
            if not torch.isfinite(gradient) or gradient <= 0:
                raise RuntimeError("nonfinite or zero correction gradient")
            optimizer.step()
            losses.append(loss.item())
            gradients.append(float(gradient))
            log.write(
                json.dumps(
                    {
                        "step": step,
                        "loss": loss.item(),
                        "gradient_norm": float(gradient),
                    }
                )
                + "\n"
            )
    if losses[-1] >= losses[0]:
        raise RuntimeError("perturbation correction loss did not decrease")
    if any(
        not torch.equal(parameter.detach(), frozen[name]) or parameter.grad is not None
        for name, parameter in model.named_parameters()
        if name in frozen
    ):
        raise RuntimeError("frozen component changed or received a gradient")
    if not torch.equal(original_evidence, evidence) or not torch.equal(
        original_condition, condition
    ):
        raise RuntimeError("original facts or vectors changed")
    if torch.equal(corrected[accepted], initial[accepted]):
        raise RuntimeError("eligible condition vector was not updated")
    report = {
        "status": "passed",
        "scope": "synthetic_configured_boundary_engineering_only",
        "formal_training": False,
        "backbone_used": False,
        "natural_error_evidence": False,
        "perturbation": perturb_report,
        "steps": len(losses),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "nonzero_corrector_gradient": min(gradients) > 0,
        "eligible_vector_changed": True,
        "missing_fact_condition_unchanged": True,
        "original_facts_unchanged": True,
        "frozen_components_unchanged_and_without_gradients": True,
    }
    torch.save(
        {
            "scope": report["scope"],
            "model": model.state_dict(),
            "initial": initial,
            "perturbed": perturbed,
            "config": config,
        },
        output_dir / "perturbation_diagnostic.pt",
    )
    (output_dir / "perturbation_diagnostic_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report
