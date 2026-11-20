"""Run staged optimization on explicit engineering fixtures.

Fixture labels are program-verified, synthetic, and are NEVER approved
ReimburseBench supervision. This verifies trainability, not generalization.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from audit_engine.models.heads import AuditHeads, parameter_count
from audit_engine.training.diagnostics import run_perturbation_diagnostic
from audit_engine.training.forward import (
    ScalarCondition,
    correction_inputs,
    forward_condition,
)
from audit_engine.training.losses import (
    masked_reference_loss,
    masked_relation_loss,
    masked_state_loss,
    state_probabilities,
)
from audit_engine.training.perturb import generate_detectable_errors
from audit_engine.training.stages import configure_trainable_stage


def fixture_spec() -> dict:
    cases = []
    for identifier, amount, missing in [
        ("fixture_a", "90", False),
        ("fixture_b", "120", False),
        ("fixture_c", "70", True),
    ]:
        cases.append(
            {
                "workflow_id": identifier,
                "amount": amount,
                "missing_budget": missing,
                "application": f"Employee Lin claims USD {amount} for one hotel night.",
                "policy": "Reimburse a hotel expense only when its amount does not exceed the approved accommodation budget. The approval must be available.",
                "evidence": [
                    f"Hotel invoice. Guest: Lin. One night. Total paid: USD {amount}.",
                    "Approved accommodation budget for Lin: USD 100 for one night.",
                    "Cafeteria lunch menu: rice and vegetables. No accommodation or budget information.",
                ],
            }
        )
    return {
        "label_method": "synthetic_program_verified_engineering_only",
        "cases": cases,
    }


def fixture_packets(spec: dict, device: str):
    policies, relations, references, states, active = [], [], [], [], []
    for case in spec["cases"]:
        policies.append(
            ScalarCondition(
                "LE", ((case["amount"], None, None), (None, "100", None)), (0, 1)
            )
        )
        relation = (
            2 if case["missing_budget"] else (0 if float(case["amount"]) <= 100 else 1)
        )
        relations.append([relation, relation, 2])
        refs = [[0.0, 0.0] for _ in range(3)]
        if relation < 2:
            refs[0][relation] = refs[1][relation] = 1.0
        references.append(refs)
        states.append(relation)
        active.append([True, not case["missing_budget"], True])
    active_tensor = torch.tensor(active, device=device)
    return (
        policies,
        {
            "relation": torch.tensor(relations, device=device),
            "reference": torch.tensor(references, device=device),
            "state": torch.tensor(states, device=device),
            "relation_mask": active_tensor,
            "reference_mask": active_tensor[..., None].expand(-1, -1, 2),
            "state_mask": torch.ones(len(policies), dtype=torch.bool, device=device),
        },
        active_tensor,
    )


def supervision(output: dict, labels: dict, stage: str):
    relation = masked_relation_loss(
        output["logits"], labels["relation"], labels["relation_mask"]
    )
    reference = masked_reference_loss(
        output["references"], labels["reference"], labels["reference_mask"]
    )
    if stage == "B1":
        return relation
    if stage == "B2":
        return reference
    if stage == "C":
        state = masked_state_loss(
            state_probabilities(output["strengths"]),
            labels["state"],
            labels["state_mask"],
        )
        return relation + reference + 0.5 * state
    return relation + reference


def run(
    features_path: Path, output_dir: Path, device: str, steps: int, config: dict
) -> dict:
    torch.manual_seed(config["project"]["seed"])
    torch.set_num_threads(4)
    output_dir.mkdir(parents=True, exist_ok=True)
    cached = torch.load(features_path, map_location="cpu", weights_only=True)
    if cached.get("label_method") != "synthetic_program_verified_engineering_only":
        raise ValueError("smoke runner only accepts explicit engineering fixtures")
    features = {key: cached[key].to(device) for key in ("raw_e", "raw_x", "raw_p")}
    spec = fixture_spec()
    policies, labels, active = fixture_packets(spec, device)
    model = AuditHeads().to(device)
    log = (output_dir / "learning_curve.jsonl").open("w")
    records = []
    initial_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    started = time.monotonic()
    # Full-batch fixtures each have one condition per workflow; equal weights.
    for stage in ("B1", "B2", "B3"):
        configure_trainable_stage(model, stage)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config["heads"]["learning_rate"],
            weight_decay=0.01,
        )
        frozen = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if not p.requires_grad
        }
        losses = []
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            evidence, condition = model.shared(
                features["raw_e"], features["raw_x"], features["raw_p"]
            )
            output = forward_condition(model, condition, evidence, policies, active)
            loss = supervision(output, labels, stage)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss in {stage}")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            losses.append(loss.item())
            log.write(
                json.dumps(
                    {
                        "stage": stage,
                        "step": step,
                        "loss": loss.item(),
                        "gradient_norm": float(grad),
                    }
                )
                + "\n"
            )
            log.flush()
        if any(
            not torch.equal(p.detach(), frozen[name])
            for name, p in model.named_parameters()
            if name in frozen
        ):
            raise RuntimeError(f"frozen parameters changed in {stage}")
        records.append(
            {
                "stage": stage,
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "steps": steps,
                "frozen_parameters_unchanged": True,
            }
        )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all()
                if device.startswith("cuda")
                else [],
                "stage": stage,
                "config": config,
                "data_signature": cached.get("signature"),
                "scope": spec["label_method"],
            },
            output_dir / f"{stage}.pt",
        )
        print(json.dumps(records[-1]), flush=True)
    with torch.no_grad():
        evidence, condition = model.shared(
            features["raw_e"], features["raw_x"], features["raw_p"]
        )
        result = forward_condition(model, condition, evidence, policies, active)
        correct = (
            (result["logits"].argmax(-1) == labels["relation"])[active]
            .float()
            .mean()
            .item()
        )
        ref_correct = (
            ((result["references"].sigmoid() >= 0.6) == labels["reference"].bool())[
                labels["reference_mask"]
            ]
            .float()
            .mean()
            .item()
        )
    configure_trainable_stage(model, "C")
    _, _, perturb_report = generate_detectable_errors(
        model,
        condition,
        evidence,
        policies,
        active,
        labels,
        config["correction"]["rho_relative"],
        config["thresholds"],
    )
    # Test C on the initial randomly initialized B modules. Their runtime errors
    # are label-free and naturally occur in this engineering initialization.
    # This is a gradient/optimization diagnostic, not natural-error evaluation.
    diagnostic = AuditHeads().to(device)
    diagnostic.load_state_dict(initial_state)
    configure_trainable_stage(diagnostic, "C")
    frozen = {
        name: p.detach().clone()
        for name, p in diagnostic.named_parameters()
        if not p.requires_grad
    }
    evidence, condition = diagnostic.shared(
        features["raw_e"], features["raw_x"], features["raw_p"]
    )
    before = forward_condition(diagnostic, condition, evidence, policies, active)
    errors, fact, slots, eligible = correction_inputs(
        evidence, before, active, policies, config["thresholds"]
    )
    if not bool(eligible.any()):
        raise RuntimeError("no detected eligible condition for correction diagnostic")
    optimizer = torch.optim.AdamW(
        diagnostic.corrector.parameters(), lr=config["correction"]["learning_rate"]
    )
    correction_losses = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        # Rebuild continuous context each step, retaining gradients across rounds.
        output = forward_condition(diagnostic, condition, evidence, policies, active)
        corrected = condition
        for _ in range(config["correction"]["max_rounds"]):
            errors, fact, slots, eligible_round = correction_inputs(
                evidence, output, active, policies, config["thresholds"]
            )
            corrected = diagnostic.corrector(
                corrected, output["context"], errors, fact, slots, eligible_round
            )
            output = forward_condition(
                diagnostic, corrected, evidence, policies, active
            )
        if not torch.equal(corrected[~eligible], condition[~eligible]):
            raise RuntimeError("ineligible condition changed during correction")
        loss = supervision(output, labels, "C")
        if not loss.requires_grad:
            continue
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(diagnostic.corrector.parameters(), 1.0)
        optimizer.step()
        correction_losses.append(loss.item())
        log.write(
            json.dumps(
                {
                    "stage": "C_diagnostic",
                    "step": step,
                    "loss": loss.item(),
                    "gradient_norm": float(grad),
                }
            )
            + "\n"
        )
        log.flush()
    if not correction_losses or any(
        not torch.equal(p.detach(), frozen[name])
        for name, p in diagnostic.named_parameters()
        if name in frozen
    ):
        raise RuntimeError("correction gradient or freezing verification failed")
    records.append(
        {
            "stage": "C_diagnostic_initialization_errors",
            "initial_loss": correction_losses[0],
            "final_loss": correction_losses[-1],
            "steps": len(correction_losses),
            "eligible_count": int(eligible.sum()),
            "frozen_parameters_unchanged": True,
            "missing_material_bypassed": not bool(eligible[-1]),
        }
    )
    torch.save(
        {
            "model": diagnostic.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "scope": "initialization_error_diagnostic_only",
        },
        output_dir / "C_diagnostic.pt",
    )
    log.close()
    controlled_perturbation = run_perturbation_diagnostic(
        output_dir / "controlled_perturbation", device, steps, config
    )
    report = {
        "bounded_perturbation": perturb_report,
        "controlled_perturbation_diagnostic": controlled_perturbation,
        "scope": spec["label_method"],
        "formal_training": False,
        "features_source": cached.get("signature"),
        "device": device,
        "parameters": parameter_count(model),
        "stages": records,
        "fixture_relation_accuracy": correct,
        "fixture_reference_accuracy": ref_correct,
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3
        if device.startswith("cuda")
        else None,
        "all_stage_losses_decreased": all(
            r["final_loss"] < r["initial_loss"] for r in records
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/smoke/heads"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--config", type=Path, default=Path("configs/base512.yaml"))
    args = parser.parse_args()
    report = run(
        args.features,
        args.output_dir,
        args.device,
        args.steps,
        yaml.safe_load(args.config.read_text()),
    )
    if not report["all_stage_losses_decreased"]:
        raise SystemExit("stage optimization failed")


if __name__ == "__main__":
    main()
