"""Repair normative-source reference labels and fine-tune only the citation head."""

import copy
import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from audit_engine.contracts import AuditCase, ConditionAnnotation
from audit_engine.inference.runner import citation_masks, tensors_for_case
from audit_engine.models.heads import AuditHeads
from audit_engine.training.forward import forward_condition
from audit_engine.training.losses import masked_reference_loss
from audit_engine.training.patent import feature_path, label_tensors, load_dataset, load_features, read_jsonl, validate_annotations
from audit_engine.training.stages import configure_trainable_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/patent_reference_fix_20261027_1000"))
    parser.add_argument("--delta-step-multiplier", type=float, default=1.0)
    args = parser.parse_args()
    if not 0 < args.delta_step_multiplier <= 100:
        parser.error("delta step multiplier must be in (0, 100]")
    root = args.output
    previous = Path("runs/patent_confirmed_gap_20261027_0947")
    torch.set_num_threads(4)
    torch.manual_seed(3407)
    status = {"pid": os.getpid(), "final_delivery_closed": False,
              "fine_tuned_components": ["contribution"], "held_out_answers_used_for_training": False}

    def save(**values):
        status.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        tmp = root / "status.tmp"
        tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2))
        tmp.replace(root / "status.json")

    try:
        cases = {r["workflow_id"]: AuditCase.model_validate(r) for r in read_jsonl(previous / "dataset/reader_cases.jsonl")}
        changed = {r["workflow_id"]: AuditCase.model_validate(r) for r in read_jsonl(root / "validation_cases.jsonl")}
        cases.update(changed)
        parts = {r["workflow_id"]: r for r in read_jsonl(previous / "dataset/partitions.jsonl")}
        annotations, modifications = [], []
        for row in read_jsonl(previous / "dataset/annotations.jsonl"):
            row["case_sha256"] = cases[row["workflow_id"]].digest()
            if row["status"] in {"approved", "corrected"} and row.get("state") in {"satisfied", "failed"}:
                condition = next(c for c in cases[row["workflow_id"]].conditions if c.condition_id == row["condition_id"])
                sources = sorted({ref.evidence_id for ref in condition.policy_refs})
                old_refs = copy.deepcopy(row["references"])
                row["references"] = [r for r in row["references"] if r["evidence_id"] not in sources]
                polarity = "support" if row["state"] == "satisfied" else "conflict"
                for identifier in sources:
                    for name in ("support", "conflict"):
                        row["references"].append({"evidence_id": identifier, "polarity": name,
                                                  "label": int(name == polarity), "approved": True})
                row["method"] += ":source_normative_reference_inspection_v1"
                modifications.append({"workflow_id": row["workflow_id"], "condition_id": row["condition_id"],
                    "source_policy_refs": [r.model_dump() for r in condition.policy_refs],
                    "prior_references": old_refs, "new_references": row["references"],
                    "relation_labels_unchanged": True, "reason": "A verified normative source supplies the criterion of this reviewed support/conflict judgment. Its reference label is separate from its per-file factual relation class."})
            annotations.append(ConditionAnnotation.model_validate(row))
        validate_annotations(list(cases.values()), annotations, parts)
        dataset_root = root / "dataset"
        dataset_root.mkdir(exist_ok=True)
        inputs, labels, partitions = (dataset_root / name for name in ("reader_cases.jsonl", "annotations.jsonl", "partitions.jsonl"))
        inputs.write_text("".join(c.model_dump_json() + "\n" for c in cases.values()))
        labels.write_text("".join(a.model_dump_json() + "\n" for a in annotations))
        partitions.write_text("".join(json.dumps(row) + "\n" for row in parts.values()))
        (root / "reference_label_inspection.json").write_text(json.dumps(modifications, ensure_ascii=False, indent=2))
        features = root / "features"
        features.mkdir(exist_ok=True)
        old_cases = {r["workflow_id"]: AuditCase.model_validate(r) for r in read_jsonl(previous / "dataset/reader_cases.jsonl")}
        for key, case in cases.items():
            old = old_cases[key]
            if (old.application, old.evidence, old.conditions) != (case.application, case.evidence, case.conditions):
                raise ValueError("changed neural encoding inputs require fresh encoding")
            feature = load_features(feature_path(previous / "features", key))
            feature.validate(old, 4096)
            feature.case_sha256 = case.digest()
            feature.validate(case, 4096)
            torch.save(asdict(feature), feature_path(features, key))
        selection = json.loads((previous / "source_validation_selection.json").read_text())
        checkpoint = Path(selection["selected"]["checkpoint"])
        bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
        metadata = copy.deepcopy(bundle["metadata"])
        assert metadata["three_components_optimized"]
        config = metadata["config"]
        config["data"]["partitions"] = str(partitions)
        c = config["model"]
        model = AuditHeads(c["hidden_size"], c["dimension"], c["width"], c["role_slots"])
        model.load_state_dict(bundle["model"], strict=True)
        configure_trainable_stage(model, "B2")
        dataset = load_dataset(inputs, labels, features, partitions)
        cached = {}
        save(stage="precomputing_frozen_relation_and_masking")
        for split in ("train", "validation"):
            cached[split] = []
            for item in dataset[split]:
                with torch.no_grad():
                    h, e, policies, active = tensors_for_case(model, item.case, item.features)
                    output = forward_condition(model, h, e, policies, active)
                cached[split].append((output["context"], output["probabilities"], output["delta"],
                    label_tensors(item, "cpu"), citation_masks(item.case, "cpu"), output))
        # ContributionModule consumes the fused per-file representation, which
        # is obtained by the same frozen relation computation as runtime.
        for split in cached:
            for index, item in enumerate(dataset[split]):
                with torch.no_grad():
                    h, e, policies, active = tensors_for_case(model, item.case, item.features)
                    rebuilt = [p.rebuild(a, c["role_slots"]) for p, a in zip(policies, active)]
                    role_map, _, comparison, complete = [torch.stack([r[j] for r in rebuilt]) for j in range(4)]
                    context = model.relation.context(e, role_map, active)
                    fused, _, _ = model.relation(h, e, context, comparison, complete)
                _, probabilities, delta, target, verified, _ = cached[split][index]
                cached[split][index] = (fused, probabilities, delta, target, verified)

        def measured():
            tp = fp = fn = 0
            losses = []
            with torch.no_grad():
                for fused, probabilities, delta, targets, verified in cached["validation"]:
                    logits = model.contribution(fused, probabilities, delta)
                    loss = masked_reference_loss(logits, targets["reference"], targets["reference_mask"])
                    if loss is not None:
                        losses.append(float(loss))
                    selected = (logits.sigmoid() >= config["thresholds"]["reference"]) & verified
                    known, desired = targets["reference_mask"], targets["reference"].bool()
                    tp += int((selected & desired & known).sum())
                    fp += int((selected & ~desired & known).sum())
                    fn += int((~selected & desired & known).sum())
            precision, recall = tp / (tp + fp) if tp + fp else 0, tp / (tp + fn) if tp + fn else 0
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
            return {"f1": f1, "precision": precision, "recall": recall,
                    "loss": sum(losses) / len(losses), "tp": tp, "fp": fp, "fn": fn}

        baseline = measured()
        key = lambda metrics: (-metrics["f1"], -metrics["recall"], metrics["loss"])
        best_key, best_weights, best_metrics = key(baseline), copy.deepcopy(model.state_dict()), baseline
        optimizer = torch.optim.AdamW(model.contribution.parameters(), lr=0.001, weight_decay=0.01)
        history, steps, stale, selected_epoch = [], 0, 0, 0
        save(stage="fine_tuning_reference_head", baseline=baseline)
        for epoch in range(1, 17):
            for index in torch.randperm(len(cached["train"])).tolist():
                fused, probabilities, delta, targets, _ = cached["train"][index]
                optimizer.zero_grad(set_to_none=True)
                loss = masked_reference_loss(model.contribution(fused, probabilities, delta), targets["reference"], targets["reference_mask"])
                if loss is None:
                    continue
                loss.backward()
                assert any(p.grad is not None and p.grad.ne(0).any() for p in model.contribution.parameters())
                assert all(p.grad is None for name, p in model.named_parameters() if not name.startswith("contribution."))
                torch.nn.utils.clip_grad_norm_(model.contribution.parameters(), 1.0, error_if_nonfinite=True)
                delta_columns = [(head.weight, head.weight[:, -1].detach().clone())
                                 for head in (model.contribution.support, model.contribution.conflict)]
                optimizer.step()
                # Per-coordinate learning-rate scaling for the explicit delta
                # input; the update direction still comes from TRAIN loss.
                with torch.no_grad():
                    for weight, before in delta_columns:
                        weight[:, -1] = before + args.delta_step_multiplier * (weight[:, -1] - before)
                steps += 1
            metrics = measured()
            history.append({"epoch": epoch, **metrics, "optimizer_steps": steps})
            print(json.dumps(history[-1]), flush=True)
            if key(metrics) < best_key:
                best_key, best_weights, best_metrics = key(metrics), copy.deepcopy(model.state_dict()), metrics
                selected_epoch, stale = epoch, 0
            else:
                stale += 1
                if stale >= 4:
                    break
        model.load_state_dict(best_weights)
        details = {"initial_checkpoint": str(checkpoint), "initial_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "optimizer_steps": steps, "selected_epoch": selected_epoch, "learning_rate": 0.001,
            "delta_step_multiplier": args.delta_step_multiplier,
            "baseline_source_validation": baseline, "selected_source_validation": best_metrics,
            "source_annotation_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(), "history": history,
            "relation_and_corrector_retained": True, "held_out_answers_used": False}
        metadata["reference_fine_tuning"] = details
        metadata["case_digests"] = {key: case.digest() for key, case in cases.items()}
        metadata["stage_steps"]["B2"] += steps
        metadata["nonzero_gradient_steps"]["contribution"] += steps
        weights = root / "weights"
        weights.mkdir(exist_ok=True)
        torch.save({"model": model.state_dict(), "metadata": metadata}, weights / "audit_heads.pt")
        (weights / "fine_tuning_summary.json").write_text(json.dumps(details, ensure_ascii=False, indent=2))
        save(stage="evaluating_fixed_23_workflows", reference_validation=best_metrics)
        def run(arguments):
            subprocess.run([sys.executable, *map(str, arguments)], check=True)
        predictions = root / "predictions.jsonl"
        run(["code/scripts/patent_pipeline.py", "predict", "--cases", root / "validation_cases.jsonl", "--features", features,
             "--checkpoint", weights / "audit_heads.pt", "--output", predictions, "--device", "cpu"])
        rows = list(read_jsonl(predictions))
        if len(rows) != 23 or any(row.get("status") == "failed" for row in rows):
            raise ValueError("incomplete full validation inference")
        save(stage="scoring_frozen_predictions", predictions_sha256=hashlib.sha256(predictions.read_bytes()).hexdigest())
        run(["code/scripts/evaluate_patent_predictions.py", "--predictions", predictions, "--gold", "data/labels_private/validation_reference_targets.jsonl", "--output", root / "score.json"])
        score = json.loads((root / "score.json").read_text())
        save(stage="evaluation_complete", correct=score["correct"], total=score["total"], accuracy=score["accuracy"], accuracy_target_met=score["accuracy"] > 0.5)
    except Exception as error:
        save(stage="requires_repair", error=str(error))
        raise


if __name__ == "__main__":
    main()
