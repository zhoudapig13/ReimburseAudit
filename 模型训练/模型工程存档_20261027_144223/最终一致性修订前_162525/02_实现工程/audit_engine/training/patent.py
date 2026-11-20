"""Staged, workflow-balanced training of the three patent components."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from audit_engine.contracts import AuditCase, ConditionAnnotation
from audit_engine.inference.runner import CaseFeatures, citation_masks, tensors_for_case
from audit_engine.models.heads import AuditHeads
from audit_engine.training.forward import correction_inputs, forward_condition
from audit_engine.training.losses import (
    masked_reference_loss,
    masked_relation_loss,
    masked_state_loss,
    state_probabilities,
)
from audit_engine.training.perturb import generate_detectable_errors
from audit_engine.training.stages import configure_trainable_stage

STATES = ("satisfied", "failed", "insufficient", "contradictory")
RELATIONS = ("support", "conflict", "unknown")


@dataclass
class TrainingCase:
    case: AuditCase
    features: CaseFeatures
    annotations: list[ConditionAnnotation]


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def feature_path(root: Path, workflow_id: str) -> Path:
    # Dataset identifiers never become filesystem paths.
    return root / (hashlib.sha256(workflow_id.encode()).hexdigest() + ".pt")


def load_features(path: Path) -> CaseFeatures:
    return CaseFeatures(**torch.load(path, map_location="cpu", weights_only=True))


def validate_annotations(cases_list, records, partitions=None):
    cases = {c.workflow_id: c for c in cases_list}
    if len(cases) != len(cases_list):
        raise ValueError("duplicate workflow")
    annotations = {}
    families, splits, seen = {}, {}, set()
    for a in records:
        c = cases.get(a.workflow_id)
        if c is None or a.case_sha256 != c.digest():
            raise ValueError("supervision is not bound to the current runtime case")
        if a.condition_id not in {v.condition_id for v in c.conditions}:
            raise ValueError("unknown supervised condition")
        if partitions is not None:
            partition = partitions.get(a.workflow_id)
            if not partition or (a.split, a.source_family_id) != (
                partition["split"],
                partition["source_family_id"],
            ):
                raise ValueError("annotation differs from the fixed dataset partition")
        key = (a.workflow_id, a.condition_id)
        if key in seen:
            raise ValueError("duplicate condition supervision")
        seen.add(key)
        if families.setdefault(a.source_family_id, a.split) != a.split:
            raise ValueError("source-family leakage across splits")
        if splits.setdefault(a.workflow_id, a.split) != a.split:
            raise ValueError("workflow appears in multiple splits")
        annotations.setdefault(a.workflow_id, []).append(a)
    return annotations, splits


def load_dataset(
    cases_path: Path,
    annotations_path: Path,
    features_root: Path,
    partitions_path: Path | None = None,
) -> dict[str, list[TrainingCase]]:
    cases_list = [AuditCase.model_validate(row) for row in read_jsonl(cases_path)]
    cases = {c.workflow_id: c for c in cases_list}
    partitions = (
        {v["workflow_id"]: v for v in read_jsonl(partitions_path)}
        if partitions_path
        else None
    )
    records = [
        ConditionAnnotation.model_validate(row) for row in read_jsonl(annotations_path)
    ]
    annotations, splits = validate_annotations(cases_list, records, partitions)
    result = {"train": [], "validation": [], "test": []}
    for workflow_id, records in annotations.items():
        if splits[workflow_id] == "test":
            continue
        if not any(a.status in {"approved", "corrected"} for a in records):
            continue
        case = cases[workflow_id]
        features = load_features(feature_path(features_root, workflow_id))
        result[splits[workflow_id]].append(TrainingCase(case, features, records))
    return result


def label_tensors(item: TrainingCase, device):
    conditions = {c.condition_id: i for i, c in enumerate(item.case.conditions)}
    evidence = {e.evidence_id: i for i, e in enumerate(item.case.evidence)}
    n, m = len(conditions), len(evidence)
    labels = {
        "relation": torch.full((n, m), -100, dtype=torch.long, device=device),
        "relation_mask": torch.zeros(n, m, dtype=torch.bool, device=device),
        "reference": torch.zeros(n, m, 2, device=device),
        "reference_mask": torch.zeros(n, m, 2, dtype=torch.bool, device=device),
        "state": torch.full((n,), -100, dtype=torch.long, device=device),
        "state_mask": torch.zeros(n, dtype=torch.bool, device=device),
    }
    for a in item.annotations:
        if a.status not in {"approved", "corrected"}:
            continue
        i = conditions[a.condition_id]
        if a.state is not None:
            labels["state"][i] = STATES.index(a.state)
            labels["state_mask"][i] = True
        seen_rel, seen_ref = set(), set()
        for rel in a.relations:
            if rel.evidence_id not in evidence or rel.evidence_id in seen_rel:
                raise ValueError("invalid or duplicate relation target")
            seen_rel.add(rel.evidence_id)
            if rel.approved:
                j = evidence[rel.evidence_id]
                labels["relation"][i, j] = RELATIONS.index(rel.relation)
                labels["relation_mask"][i, j] = True
        for ref in a.references:
            key = (ref.evidence_id, ref.polarity)
            if ref.evidence_id not in evidence or key in seen_ref:
                raise ValueError("invalid or duplicate reference target")
            seen_ref.add(key)
            if ref.approved:
                j, k = (
                    evidence[ref.evidence_id],
                    ("support", "conflict").index(ref.polarity),
                )
                labels["reference"][i, j, k] = ref.label
                labels["reference_mask"][i, j, k] = True
    return labels


def component_loss(output, labels, stage, relation_class_weights=None):
    terms = []
    if stage != "B2":
        terms.append(
            masked_relation_loss(
                output["logits"], labels["relation"], labels["relation_mask"],
                class_weights=relation_class_weights,
            )
        )
    if stage != "B1":
        terms.append(
            masked_reference_loss(
                output["references"], labels["reference"], labels["reference_mask"]
            )
        )
    if stage in {"C", "C2"}:
        state = masked_state_loss(
            state_probabilities(output["strengths"]),
            labels["state"],
            labels["state_mask"],
        )
        terms.append(None if state is None else 0.5 * state)
    terms = [v for v in terms if v is not None]
    return sum(terms) if terms else None


def workflow_loss(model, item, stage, config, *, training):
    h, e, policies, active = tensors_for_case(model, item.case, item.features)
    labels = label_tensors(item, h.device)
    relation_class_weights = config.get("heads", {}).get("relation_class_weights")
    if relation_class_weights is not None:
        relation_class_weights = torch.as_tensor(relation_class_weights, device=h.device, dtype=h.dtype)
    out = forward_condition(model, h, e, policies, active)
    initial_loss = component_loss(out, labels, "B3", relation_class_weights) if stage == "C2" else None
    stats = {"eligible": 0, "perturbed": 0, "perturbation_candidates": 0,
             "perturbation_original_correct": 0, "perturbation_created_errors": 0}
    if stage in {"C", "C2"}:
        if stage == "C":
            h, e = h.detach(), e.detach()
        threshold = config["thresholds"]
        verified = citation_masks(item.case, h.device)
        errors, fact, slots, eligible = correction_inputs(
            e, out, active, policies, threshold, reference_verified=verified
        )
        # Natural runtime errors are the primary correction source. Gold is used
        # only to select offline perturbations, never to construct error signals.
        if (
            training
            and bool(labels["relation_mask"].any())
            and bool(labels["reference_mask"].any())
        ):
            candidate, accepted, perturbation_report = generate_detectable_errors(
                model,
                h,
                e,
                policies,
                active,
                labels,
                config["correction"]["rho_relative"],
                threshold,
                reference_verified=verified,
            )
            stats["perturbation_candidates"] = perturbation_report["candidate_count"]
            stats["perturbation_original_correct"] = perturbation_report["original_correct_count"]
            stats["perturbation_created_errors"] = perturbation_report["created_error_count"]
            h = torch.where(accepted[:, None], candidate, h)
            stats["perturbed"] = int(accepted.sum())
            out = forward_condition(model, h, e, policies, active)
            errors, fact, slots, eligible = correction_inputs(
                e, out, active, policies, threshold, reference_verified=verified
            )
        stats["eligible"] = int(eligible.sum())
        if not bool(eligible.any()):
            return (0.1 * initial_loss if initial_loss is not None else None), stats
        for key in ("relation_mask", "reference_mask", "state_mask"):
            labels[key] &= eligible.reshape((-1,) + (1,) * (labels[key].ndim - 1))
        rounds = config["correction"].get("max_rounds", 2)
        if not 1 <= rounds <= 2:
            raise ValueError("correction training unfolds one or two rounds")
        for round_index in range(rounds):
            if round_index:
                errors, fact, slots, eligible = correction_inputs(
                    e, out, active, policies, threshold, reference_verified=verified
                )
                if not bool(eligible.any()):
                    break
            h = model.corrector(h, out["context"], errors, fact, slots, eligible)
            out = forward_condition(model, h, e, policies, active)
    loss = component_loss(out, labels, stage, relation_class_weights)
    if initial_loss is not None:
        loss = 0.1 * initial_loss if loss is None else loss + 0.1 * initial_loss
    return loss, stats


def validate(model, items, stage, config):
    model.eval()
    values = []
    with torch.no_grad():
        for item in items:
            loss, _ = workflow_loss(model, item, stage, config, training=False)
            if loss is not None:
                values.append(float(loss))
    return sum(values) / len(values) if values else None


@torch.no_grad()
def component_validation(model, items, config):
    state_confusion = torch.zeros(4, 4, dtype=torch.long)
    relation_confusion = torch.zeros(3, 3, dtype=torch.long)
    tp = fp = fn = 0
    for item in items:
        h, e, policies, active = tensors_for_case(model, item.case, item.features)
        output = forward_condition(model, h, e, policies, active)
        labels = label_tensors(item, h.device)
        _, facts, _, _ = correction_inputs(
            e,
            output,
            active,
            policies,
            config["thresholds"],
            reference_verified=citation_masks(item.case, h.device),
        )
        for target, prediction, approved, confusion in (
            (
                labels["state"],
                facts[:, -4:].argmax(-1),
                labels["state_mask"],
                state_confusion,
            ),
            (
                labels["relation"],
                output["logits"].argmax(-1),
                labels["relation_mask"],
                relation_confusion,
            ),
        ):
            size = confusion.size(0)
            encoded = target[approved] * size + prediction[approved]
            confusion += torch.bincount(encoded.cpu(), minlength=size * size).reshape(
                size, size
            )
        selected = output["references"].sigmoid() >= config["thresholds"]["reference"]
        known, target = labels["reference_mask"], labels["reference"].bool()
        tp += int((selected & target & known).sum())
        fp += int((selected & ~target & known).sum())
        fn += int((~selected & target & known).sum())

    def macro(confusion):
        if not int(confusion.sum()):
            return None
        diagonal = confusion.diag().float()
        denominator = confusion.sum(0) + confusion.sum(1)
        return float((2 * diagonal / denominator.clamp_min(1)).mean())

    return {
        "state_macro_f1": macro(state_confusion),
        "relation_macro_f1": macro(relation_confusion),
        "reference_recall": tp / (tp + fn) if tp + fn else None,
        "reference_false_positive_fraction": fp / (tp + fp) if tp + fp else 0.0,
        "approved_states": int(state_confusion.sum()),
        "approved_relations": int(relation_confusion.sum()),
    }


@torch.no_grad()
def correction_validation(model, items, config):
    """Select C on natural errors; report corrections and unwanted changes."""
    fixed, spoiled, supervised, rates = 0, 0, 0, []
    for item in items:
        h, e, policies, active = tensors_for_case(model, item.case, item.features)
        labels = label_tensors(item, h.device)
        verified = citation_masks(item.case, h.device)
        before = forward_condition(model, h, e, policies, active)
        errors, fact, slots, eligible = correction_inputs(
            e,
            before,
            active,
            policies,
            config["thresholds"],
            reference_verified=verified,
        )
        after = before
        for _ in range(config["correction"].get("max_rounds", 2)):
            if not bool(eligible.any()):
                break
            h = model.corrector(h, after["context"], errors, fact, slots, eligible)
            after = forward_condition(model, h, e, policies, active)
            errors, fact, slots, eligible = correction_inputs(
                e, after, active, policies, config["thresholds"], reference_verified=verified
            )

        def correctness(output):
            _, f, _, _ = correction_inputs(
                e,
                output,
                active,
                policies,
                config["thresholds"],
                reference_verified=verified,
            )
            rel = (
                (output["logits"].argmax(-1) == labels["relation"])
                | ~labels["relation_mask"]
            ).all(-1)
            ref = (
                (
                    (
                        output["references"].sigmoid()
                        >= config["thresholds"]["reference"]
                    )
                    == labels["reference"].bool()
                )
                | ~labels["reference_mask"]
            ).all(dim=(-1, -2))
            return rel & ref & (f[:, -4:].argmax(-1) == labels["state"])

        known = (
            labels["state_mask"]
            & labels["relation_mask"].any(-1)
            & labels["reference_mask"].any(dim=(-1, -2))
        )
        old, new = correctness(before), correctness(after)
        positive = int((~old & new & known).sum())
        negative = int((old & ~new & known).sum())
        count = int(known.sum())
        fixed += positive
        spoiled += negative
        supervised += count
        if count:
            rates.append((positive - negative) / count)
    return {
        "natural_errors_corrected": fixed,
        "unwanted_changes": spoiled,
        "supervised_conditions": supervised,
        "workflow_mean_net_correction": sum(rates) / len(rates) if rates else None,
    }


def train(
    model: AuditHeads, dataset, config: dict, output: Path, *, qualification="formal"
):
    if not dataset["train"] or not dataset["validation"]:
        raise ValueError("approved train and validation condition labels are required")
    if qualification == "formal":
        adapter = config["model"].get("adapter_path")
        expected_adapter = (
            hashlib.sha256(
                (Path(adapter) / "adapter_model.safetensors").read_bytes()
            ).hexdigest()
            if adapter
            else None
        )
        feature_signatures = {
            json.dumps(item.features.signature, sort_keys=True)
            for items in dataset.values()
            for item in items
        }
        if len(feature_signatures) != 1:
            raise ValueError("formal training cannot mix incompatible feature encoders")
        for items in dataset.values():
            for item in items:
                signature = item.features.signature
                if (
                    signature.get("kind") != "frozen_backbone"
                    or signature.get("model") != config["model"]["name"]
                    or signature.get("revision") != config["model"]["revision"]
                    or signature.get("adapter_sha256") != expected_adapter
                ):
                    raise ValueError(
                        "formal training requires actual frozen-backbone encodings"
                    )
                if not item.case.input_version.startswith("patent_reader_v1:"):
                    raise ValueError(
                        "formal training requires actual source-reader inputs"
                    )
                policies = {c.condition_id: c for c in item.case.conditions}
                if any(
                    a.status in {"approved", "corrected"}
                    and policies[a.condition_id].compile_status != "verified"
                    for a in item.annotations
                ):
                    raise ValueError(
                        "approved component labels require source-reviewed compiled conditions"
                    )
        minimum = config.get("data", {}).get("minimum_approved_train_workflows", 20)
        if len(dataset["train"]) < minimum:
            raise ValueError(
                f"formal pilot requires at least {minimum} source-reviewed training workflows"
            )
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config["project"]["seed"])
    stage_steps, history = {}, []
    gradient_steps = {
        name: 0 for name in ("shared", "relation", "contribution", "corrector")
    }
    for stage, epochs in config["fast_training"]["epochs"].items():
        training_items = dataset["train"]
        if stage == "B4":
            augmented_items = [
                i
                for i in training_items
                if any(a.augmentation_of for a in i.annotations)
            ]
            if not augmented_items:
                stage_steps[stage] = 0
                history.append(
                    {
                        "stage": stage,
                        "status": "skipped",
                        "reason": "no_source_verified_factual_augmentations",
                    }
                )
                continue
        elif stage in {"B1", "B2", "B3"}:
            training_items = [i for i in training_items if not any(a.augmentation_of for a in i.annotations)]
        configure_trainable_stage(model, stage)
        rate = (config["correction"]["learning_rate"] if stage == "C"
                else config["heads"]["learning_rate"])
        multiplier = config["heads"].get("policy_learning_rate_multiplier", 1.0)
        if not 0 < multiplier <= 100:
            raise ValueError("policy learning-rate multiplier must be in (0, 100]")
        policy_parameters = list(model.relation.policy.parameters())
        policy_ids = {id(parameter) for parameter in policy_parameters}
        groups = [{"params": [parameter for parameter in model.parameters()
                              if parameter.requires_grad and id(parameter) not in policy_ids],
                   "lr": rate},
                  {"params": [parameter for parameter in policy_parameters if parameter.requires_grad],
                   "lr": rate * multiplier}]
        optimizer = torch.optim.AdamW(
            [group for group in groups if group["params"]],
            lr=rate, weight_decay=config["heads"]["weight_decay"],
        )
        # A warm start can outperform every attempted update. Include its
        # validation result in selection instead of unconditionally replacing
        # it with the first epoch.
        baseline_loss = validate(model, dataset["validation"], stage, config)
        baseline_component = component_validation(model, dataset["validation"], config)
        baseline_correction = None
        if stage in {"C", "C2"}:
            baseline_correction = correction_validation(model, dataset["validation"], config)
            net = baseline_correction["workflow_mean_net_correction"]
            best = None if net is None else (
                -net, baseline_correction["unwanted_changes"],
                baseline_loss if baseline_loss is not None else 1e30,
            )
        else:
            f1 = baseline_component["state_macro_f1"]
            best = None if f1 is None else (
                -f1, -(baseline_component["reference_recall"] or 0.0),
                baseline_component["reference_false_positive_fraction"],
                baseline_loss if baseline_loss is not None else 1e30,
            )
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()} if best is not None else None
        best_optimizer = copy.deepcopy(optimizer.state_dict()) if best is not None else None
        selected_epoch, stale, steps = 0, 0, 0
        history.append({"stage": stage, "epoch": 0, "status": "selection_baseline",
                        "validation_loss": baseline_loss, "selection_score": best,
                        "component_validation": baseline_component,
                        "correction_validation": baseline_correction})
        for epoch in range(epochs):
            configure_trainable_stage(model, stage)
            order = torch.randperm(len(training_items)).tolist()
            total, count, eligible, perturbed = 0.0, 0, 0, 0
            perturbation_counts = {name: 0 for name in (
                "perturbation_candidates", "perturbation_original_correct",
                "perturbation_created_errors")}
            # Exactly one mean loss per workflow. No second inverse-workflow weight.
            for index in order:
                optimizer.zero_grad(set_to_none=True)
                loss, stats = workflow_loss(
                    model, training_items[index], stage, config, training=True
                )
                eligible += stats["eligible"]
                perturbed += stats["perturbed"]
                for name in perturbation_counts:
                    perturbation_counts[name] += stats[name]
                if loss is None:
                    continue
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite training loss")
                loss.backward()
                for name in gradient_steps:
                    if any(
                        p.grad is not None and bool(p.grad.detach().ne(0).any())
                        for p in getattr(model, name).parameters()
                    ):
                        gradient_steps[name] += 1
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config["heads"]["gradient_clip"],
                    error_if_nonfinite=True,
                )
                optimizer.step()
                steps += 1
                total += float(loss.detach())
                count += 1
            score = validate(model, dataset["validation"], stage, config)
            component_metrics = component_validation(
                model, dataset["validation"], config
            )
            f1 = component_metrics["state_macro_f1"]
            selection = (
                None
                if f1 is None
                else (
                    -f1,
                    -(component_metrics["reference_recall"] or 0.0),
                    component_metrics["reference_false_positive_fraction"],
                    score if score is not None else 1e30,
                )
            )
            correction_metrics = None
            if stage in {"C", "C2"}:
                correction_metrics = correction_validation(
                    model, dataset["validation"], config
                )
                net = correction_metrics["workflow_mean_net_correction"]
                selection = (
                    None
                    if net is None
                    else (
                        -net,
                        correction_metrics["unwanted_changes"],
                        score if score is not None else 1e30,
                    )
                )
            row = {
                "stage": stage,
                "epoch": epoch + 1,
                "train_loss": total / count if count else None,
                "validation_loss": score,
                "optimizer_steps": steps,
                "eligible": eligible,
                "perturbed": perturbed,
                **perturbation_counts,
                "selection_score": selection,
                "correction_validation": correction_metrics,
                "component_validation": component_metrics,
            }
            history.append(row)
            with (output / "training.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            if selection is not None and (best is None or selection < best):
                best, stale = selection, 0
                selected_epoch = epoch + 1
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                best_optimizer = copy.deepcopy(optimizer.state_dict())
            else:
                stale += 1
            if stale >= config["fast_training"]["patience"]:
                break
        stage_steps[stage] = steps
        if best_state is not None:
            model.load_state_dict(best_state)
            optimizer.load_state_dict(best_optimizer)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "stage": stage,
                "stage_steps": dict(stage_steps),
                "selected_epoch": selected_epoch,
                "config": config,
                "qualification": qualification,
            },
            output / f"stage_{stage}.pt",
        )
    completed = all(stage_steps.get(k, 0) > 0 for k in ("B1", "B2", "B3", "C")) and all(
        count > 0 for count in gradient_steps.values()
    )
    status = {
        "qualification": qualification,
        "three_components_optimized": completed,
        "stage_steps": stage_steps,
        "nonzero_gradient_steps": gradient_steps,
        "config": config,
        "workflows": {k: len(v) for k, v in dataset.items()},
        "original_training_workflows": sum(not any(a.augmentation_of for a in i.annotations) for i in dataset["train"]),
        "source_verified_training_variants": sum(any(a.augmentation_of for a in i.annotations) for i in dataset["train"]),
        "case_digests": {
            i.case.workflow_id: i.case.digest()
            for rows in dataset.values()
            for i in rows
        },
        "feature_signatures": [
            i.features.signature for rows in dataset.values() for i in rows
        ],
        "history": history,
    }
    torch.save(
        {"model": model.state_dict(), "metadata": status}, output / "audit_heads.pt"
    )
    (output / "training_summary.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return status


def reader_adapter_transfer_allowed(previous_signatures, current_signatures):
    """Allow retraining after only the pinned reader adapter changed.

    This is not prediction compatibility: inference still requires the exact
    feature signature saved by the new training run.
    """
    adapter_keys = {"adapter_sha256", "adapter_config_sha256"}
    if not previous_signatures or not current_signatures:
        return False
    def core(signature):
        return {k: v for k, v in signature.items() if k not in adapter_keys}
    for current in current_signatures:
        if current.get("kind") != "frozen_backbone":
            return False
        if any(not isinstance(current.get(key), str) or len(current[key]) != 64
               for key in adapter_keys):
            return False
        if not any(core(current) == core(previous) for previous in previous_signatures):
            return False
    return True
