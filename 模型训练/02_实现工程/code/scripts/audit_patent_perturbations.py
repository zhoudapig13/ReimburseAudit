"""Measure genuine bounded, detectable internal errors on TRAIN only."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from audit_engine.inference.runner import citation_masks, tensors_for_case
from audit_engine.models.heads import AuditHeads
from audit_engine.training.patent import label_tensors, load_dataset
from audit_engine.training.perturb import generate_detectable_errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    bundle = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = bundle["metadata"]["config"]
    c = config["model"]
    model = AuditHeads(c["hidden_size"], c["dimension"], c["width"], c["role_slots"])
    model.load_state_dict(bundle["model"], strict=True)
    model.eval().requires_grad_(False)
    root = args.dataset_run
    dataset = load_dataset(root / "dataset/reader_cases.jsonl", root / "dataset/annotations.jsonl",
                           root / "features", root / "dataset/partitions.jsonl")
    reports, accepted_cases = [], []
    for item in dataset["train"]:
        with torch.no_grad():
            h, e, policies, active = tensors_for_case(model, item.case, item.features)
        labels = label_tensors(item, "cpu")
        if not labels["relation_mask"].any() or not labels["reference_mask"].any():
            continue
        for radius in (0.01, 0.03, 0.05):
            perturbed, accepted, report = generate_detectable_errors(
                model, h, e, policies, active, labels, radius, config["thresholds"],
                reference_verified=citation_masks(item.case, "cpu"))
            reports.append({"workflow_id": item.case.workflow_id, **report})
            if accepted.any():
                accepted_cases.append({"workflow_id": item.case.workflow_id,
                    "case_sha256": item.case.digest(), "rho_relative": radius,
                    "condition_ids": [c.condition_id for c in item.case.conditions],
                    "accepted": accepted, "perturbed_condition": perturbed})
    assert not any(parameter.grad is not None for parameter in model.parameters())
    torch.save(accepted_cases, args.output / "accepted_training_perturbations.pt")
    summary = {"selection_split": "train", "source_facts_changed": False,
               "model_parameters_changed": False, "reference_answers_read": False,
               "checkpoint": str(args.checkpoint), "by_radius": {
                   str(radius): {key: sum(r[key] for r in reports if r["rho_relative"] == radius)
                       for key in ("candidate_count", "original_correct_count", "created_error_count",
                                   "accepted_detectable_count")}
                   for radius in (0.01, 0.03, 0.05)}, "reports": reports}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "reports"}))


if __name__ == "__main__":
    main()
