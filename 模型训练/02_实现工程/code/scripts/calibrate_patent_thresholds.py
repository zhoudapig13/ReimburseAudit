"""Calibrate thresholds on approved condition labels; score benchmark only afterwards."""

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from audit_engine.inference.runner import citation_masks, tensors_for_case
from audit_engine.models.heads import AuditHeads
from audit_engine.training.forward import correction_inputs, forward_condition
from audit_engine.training.patent import label_tensors, load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = bundle["metadata"]["config"]
    c = config["model"]
    model = AuditHeads(c["hidden_size"], c["dimension"], c["width"], c["role_slots"])
    model.load_state_dict(bundle["model"], strict=True)
    model.eval()
    source = args.dataset_run
    dataset = load_dataset(source / "dataset/reader_cases.jsonl", source / "dataset/annotations.jsonl",
                           source / "features", source / "dataset/partitions.jsonl")
    cached = []
    with torch.no_grad():
        for item in dataset["validation"]:
            h, e, policies, active = tensors_for_case(model, item.case, item.features)
            output = forward_condition(model, h, e, policies, active)
            cached.append((e, output, active, policies, citation_masks(item.case, "cpu"), label_tensors(item, "cpu")))

    def measured(thresholds):
        confusion = torch.zeros(4, 4, dtype=torch.long)
        tp = fp = fn = 0
        for e, output, active, policies, verified, labels in cached:
            _, facts, _, _ = correction_inputs(e, output, active, policies, thresholds,
                                              reference_verified=verified)
            mask = labels["state_mask"]
            indices = labels["state"][mask] * 4 + facts[:, -4:].argmax(-1)[mask]
            confusion += torch.bincount(indices, minlength=16).reshape(4, 4)
            chosen = (output["references"].sigmoid() >= thresholds["reference"]) & verified
            desired, known = labels["reference"].bool(), labels["reference_mask"]
            tp += int((chosen & desired & known).sum())
            fp += int((chosen & ~desired & known).sum())
            fn += int((~chosen & desired & known).sum())
        denominator = confusion.sum(0) + confusion.sum(1)
        present = denominator > 0
        macro = ((2 * confusion.diag()[present]) / denominator[present]).mean().item()
        return {"state_macro_f1": macro, "state_confusion": confusion.tolist(),
                "reference_f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0,
                "reference_precision": tp / (tp + fp) if tp + fp else 0,
                "reference_recall": tp / (tp + fn) if tp + fn else 0,
                "approved_states": int(confusion.sum())}

    baseline = measured(config["thresholds"])
    candidates = []
    for support, conflict, reference in itertools.product((0.4, 0.55, 0.7), (0.35, 0.5, 0.65, 0.7), (0.25, 0.4, 0.6)):
        thresholds = dict(config["thresholds"], support=support, conflict=conflict, reference=reference)
        candidates.append({"thresholds": thresholds, **measured(thresholds)})
    best = max(candidates, key=lambda r: (r["state_macro_f1"], r["reference_f1"], r["reference_precision"]))
    original = {"thresholds": config["thresholds"], **baseline}
    if (baseline["state_macro_f1"], baseline["reference_f1"], baseline["reference_precision"]) >= (
        best["state_macro_f1"], best["reference_f1"], best["reference_precision"]
    ):
        best = original
    metadata = copy.deepcopy(bundle["metadata"])
    metadata["config"]["thresholds"] = best["thresholds"]
    report = {"checkpoint": str(args.checkpoint), "selection": "approved_validation_condition_labels",
              "benchmark_targets_read": False, "baseline": original, "selected": best,
              "candidate_count": len(candidates), "candidates": candidates}
    metadata["threshold_calibration"] = {k: v for k, v in report.items() if k != "candidates"}
    torch.save({"model": bundle["model"], "metadata": metadata}, args.output / "audit_heads.pt")
    (args.output / "calibration.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "candidates"}))


if __name__ == "__main__":
    main()
