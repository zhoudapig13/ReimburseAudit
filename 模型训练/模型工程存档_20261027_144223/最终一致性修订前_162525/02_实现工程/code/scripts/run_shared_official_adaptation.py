"""Finish source reads, re-encode one shared adapter, train and evaluate 23 cases."""

import hashlib
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml

from audit_engine.contracts import AuditCase, ConditionAnnotation
from audit_engine.policy.allocations import allocate_receipt_units
from audit_engine.training.patent import read_jsonl, validate_annotations
from audit_engine.readers.text_fields import read_original_text_field
from audit_engine.inference.runner import scalar_policies
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/patent_shared_official_20261027_0822"))
    parser.add_argument("--reader-jobs", type=Path, default=Path("runs/patent_adaptation_20261027/shared_official_reader_retry_processes.json"))
    parser.add_argument("--compiled-cases", type=Path, default=Path("data/patent_reviewed/policy_cases_v13.jsonl"))
    parser.add_argument("--reader-roots", nargs="+", default=["data/patent_shared_official_reader_train", "data/patent_shared_official_reader_validation", "data/patent_shared_official_reader_remaining"])
    parser.add_argument("--train-selection", type=Path, default=Path("runs/patent_adaptation_20261027/new_shared_train_selection.json"))
    parser.add_argument("--train-workflows", type=int, default=20)
    parser.add_argument("--reviews", type=Path, default=Path("data/patent_source_review_v13"))
    parser.add_argument("--config", type=Path, default=Path("configs/patent_shared_official_reader.yaml"))
    parser.add_argument("--extra-validation-annotations", type=Path)
    parser.add_argument("--bind-text-fields", action="store_true")
    parser.add_argument("--refine-policy-references", action="store_true")
    parser.add_argument("--initial-projection-gain", type=float, default=1.0)
    parser.add_argument("--variant-prefix", default="patent_variants_shared_official_0822")
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    status = {"pid": os.getpid(), "final_delivery_closed": False}

    def save(**values):
        status.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        temporary = root / "status.tmp"
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2))
        temporary.replace(root / "status.json")

    def start(arguments, name, gpu=None):
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with (root / (name + ".log")).open("ab") as log:
            child = subprocess.Popen([sys.executable, "-u", *map(str, arguments)],
                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        return child, {"pid": child.pid, "name": name, "gpu": gpu}

    def run(arguments, name):
        child, job = start(arguments, name)
        save(job=job)
        if child.wait():
            raise RuntimeError(name + " failed; inspect its saved log")

    try:
        jobs = json.loads(args.reader_jobs.read_text())
        save(stage="waiting_for_actual_source_readers", reader_jobs=jobs)
        while any(Path("/proc/" + str(job["pid"])).exists() for job in jobs):
            time.sleep(10)
        compiled = {row["workflow_id"]: AuditCase.model_validate(row)
                    for row in read_jsonl(args.compiled_cases)}
        partitions = {row["workflow_id"]: row for row in read_jsonl(Path("data/patent_full/partitions.jsonl"))}
        train_ids = json.loads(args.train_selection.read_text())["workflow_ids"]
        validation_ids = [key for key, row in partitions.items() if row["split"] == "validation"]
        expected = set(train_ids + validation_ids)
        if len(train_ids) != args.train_workflows or len(validation_ids) != 23:
            raise ValueError("source workflow selection changed")
        actual = {}
        for directory in args.reader_roots:
            for path in Path(directory).glob("shard_*/*.json"):
                row = json.loads(path.read_text())
                identifier = row.get("workflow_id")
                if identifier not in expected or "case" not in row:
                    continue
                if row.get("source_case_sha256") != compiled[identifier].digest():
                    continue  # A separately recorded older policy input version.
                case = AuditCase.model_validate(row["case"])
                if case.digest() != row["case_sha256"]:
                    raise ValueError("actual reader record checksum changed")
                actual[identifier] = case
        if set(actual) != expected:
            raise ValueError("missing actual source-reader cases: " + str(sorted(expected - set(actual))))
        save(stage="building_source_scoped_inputs", source_workflows=len(actual))
        text_records = []
        if args.bind_text_fields:
            for identifier, case in list(actual.items()):
                if not identifier.startswith("1.2."):
                    continue
                raw = case.model_dump()
                for locator in [
                    {"evidence_id": "e001", "condition_id": "merchant_city_matches_travel", "role": "documented_e001_travel_city", "mode": "labeled_text", "label": "Meeting Location", "first_comma_part": True},
                    {"evidence_id": "e001", "condition_id": "meal_date_matches_documented_delay", "role": "documented_e001_delay_interval", "mode": "date_interval", "start_contains": "The meeting concluded promptly", "end_contains": "The meal cost was paid uniformly"}]:
                    observation, provenance = read_original_text_field(case, Path("data_raw/ReimburseBench"), **locator)
                    prior = [o for o in case.operands if o.condition_id == locator["condition_id"] and o.role == locator["role"] and o.evidence_id == locator["evidence_id"]]
                    raw["operands"] = [o for o in raw["operands"] if not (o["condition_id"] == locator["condition_id"] and o["role"] == locator["role"] and o["evidence_id"] == locator["evidence_id"])] + [observation.model_dump()]
                    provenance.update(workflow_id=identifier, prior_observations=[o.model_dump() for o in prior])
                    text_records.append(provenance)
                raw["input_version"] = "patent_reader_v1:original_text_fields:" + hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
                bound = AuditCase.model_validate(raw)
                if bound.policy_coverage == "verified" and "material_reader_failure" in bound.unresolved:
                    active = torch.ones(len(bound.evidence), dtype=torch.bool)
                    ready = True
                    for policy in scalar_policies(bound):
                        roles, required, _, complete = policy.rebuild(active, 8)
                        ready &= bool((roles.any(-1) | ~required).all())
                        ready &= bool((complete.squeeze(-1).bool() | ~roles.any(0)).all()) if policy.comparison_required else policy.semantic_operands_ready(active)
                    if ready:
                        raw["unresolved"].remove("material_reader_failure")
                        bound = AuditCase.model_validate(raw)
                actual[identifier] = bound
            (root / "text_field_provenance.json").write_text(json.dumps(text_records, ensure_ascii=False, indent=2))
        allocation_records = []
        for identifier in ("4.3.3", "4.3.4"):
            case = actual[identifier]
            receipt_ids = ["e007", "e008"] if identifier == "4.3.3" else ["e008", "e009"]
            plan = []
            for index, evidence_id in enumerate(receipt_ids):
                condition_id = ("passenger_identity_" if identifier == "4.3.3" else "not_previously_reimbursed_") + evidence_id
                plan.append({"object_id": "ticket_" + evidence_id, "evidence_id": evidence_id,
                    "amount_condition_id": condition_id, "amount_role": "actual_" + evidence_id + "_fare",
                    "condition_ids": [condition_id]})
            try:
                actual[identifier], provenance = allocate_receipt_units(case, plan)
                allocation_records.append(provenance)
            except ValueError as error:
                raw = case.model_dump()
                for condition in raw["conditions"]:
                    condition["consequence"]["on_failed"] = "manual_review"
                raw["unresolved"].append("receipt_allocation_unresolved")
                actual[identifier] = AuditCase.model_validate(raw)
                allocation_records.append({"workflow_id": identifier, "unresolved": str(error)})
        originals = root / "originals"
        originals.mkdir(exist_ok=True)
        ordered = [actual[key] for key in train_ids + validation_ids]
        (originals / "reader_cases.jsonl").write_text("".join(case.model_dump_json() + "\n" for case in ordered))
        (root / "receipt_allocation_provenance.json").write_text(json.dumps(allocation_records, ensure_ascii=False, indent=2))
        run(["code/scripts/compile_source_annotations.py", "--cases", originals / "reader_cases.jsonl",
             "--reviews", args.reviews, "--output", originals / "annotations.jsonl"], "annotations")
        labels = list(read_jsonl(originals / "annotations.jsonl"))
        for row in read_jsonl(Path("runs/patent_full_20261027/reviewed_full_v6/annotations.jsonl")):
            if row["workflow_id"] in {"4.2.4", "4.3.5"}:
                row["case_sha256"] = actual[row["workflow_id"]].digest()
                labels.append(row)
        if args.extra_validation_annotations:
            for row in read_jsonl(args.extra_validation_annotations):
                if row["split"] == "validation" and row["workflow_id"].startswith("1.2."):
                    row["case_sha256"] = actual[row["workflow_id"]].digest()
                    labels.append(row)
        if args.refine_policy_references:
            for row in labels:
                if row.get("state") not in {"satisfied", "failed"}:
                    continue
                condition = next(c for c in actual[row["workflow_id"]].conditions if c.condition_id == row["condition_id"])
                sources = {ref.evidence_id for ref in condition.policy_refs}
                row["references"] = [ref for ref in row["references"] if ref["evidence_id"] not in sources]
                polarity = "support" if row["state"] == "satisfied" else "conflict"
                row["references"].extend({"evidence_id": key, "polarity": name, "label": int(name == polarity), "approved": True}
                    for key in sorted(sources) for name in ("support", "conflict"))
                row["method"] += ":source_normative_reference_inspection_v1"
        labels = list({(row["workflow_id"], row["condition_id"]): row for row in labels}.values())
        validated = [ConditionAnnotation.model_validate(row) for row in labels]
        validate_annotations(ordered, validated, partitions)
        (originals / "annotations.jsonl").write_text("".join(row.model_dump_json() + "\n" for row in validated))
        dataset = root / "dataset"
        run(["code/scripts/build_verified_file_deletions.py", "--input-run", originals,
             "--output", dataset, "--variant-prefix", args.variant_prefix], "source_variants")
        config = yaml.safe_load(args.config.read_text())
        config["data"]["partitions"] = str(dataset / "partitions.jsonl")
        config["heads"]["initial_projection_gain"] = args.initial_projection_gain
        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        inputs = dataset / "reader_cases.jsonl"
        save(stage="encoding_current_adapter_inputs", config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
        encoding = [start(["code/scripts/patent_pipeline.py", "encode", "--cases", inputs,
                    "--features", root / "features", "--config", config_path,
                    "--model-path", "/mnt/public/zhouyibo/.cache/reimbursement-patent/model/Qwen3-VL-8B-Instruct",
                    "--shard", str(index) + "/5"], "encode_" + str(index), gpu)
                    for index, gpu in enumerate([1, 2, 3, 4, 5])]
        save(encoding_jobs=[job for _, job in encoding])
        exit_codes = [child.wait() for child, _ in encoding]
        if any(exit_codes):
            raise RuntimeError("real encoding failed: " + str(exit_codes))
        save(stage="training_all_three_components")
        run(["code/scripts/patent_pipeline.py", "train", "--cases", inputs,
             "--annotations", dataset / "annotations.jsonl", "--features", root / "features",
             "--config", config_path, "--device", "cpu", "--output", root / "weights"], "train")
        summary = json.loads((root / "weights/training_summary.json").read_text())
        if not summary["three_components_optimized"] or any(summary["stage_steps"].get(stage, 0) <= 0 for stage in ("B1", "B2", "B3", "B4", "C")):
            raise ValueError("required training stages did not all optimize parameters")
        validation_path = root / "validation_cases.jsonl"
        validation_path.write_text("".join(actual[key].model_dump_json() + "\n" for key in validation_ids))
        save(stage="predicting_complete_validation_23")
        run(["code/scripts/patent_pipeline.py", "predict", "--cases", validation_path,
             "--features", root / "features", "--checkpoint", root / "weights/audit_heads.pt",
             "--output", root / "predictions.jsonl", "--device", "cpu"], "predict")
        predictions = list(read_jsonl(root / "predictions.jsonl"))
        if len(predictions) != 23 or {row["workflow_id"] for row in predictions} != set(validation_ids):
            raise ValueError("saved predictions do not cover the fixed 23 workflows")
        save(stage="scoring_frozen_predictions", predictions_sha256=hashlib.sha256((root / "predictions.jsonl").read_bytes()).hexdigest())
        run(["code/scripts/evaluate_patent_predictions.py", "--predictions", root / "predictions.jsonl",
             "--gold", "data/labels_private/validation_reference_targets.jsonl",
             "--output", root / "score.json"], "score")
        result = json.loads((root / "score.json").read_text())
        save(stage="evaluation_complete", correct=result["correct"], total=result["total"],
             accuracy=result["accuracy"], accuracy_target_met=result["accuracy"] > 0.5)
    except Exception as error:
        save(stage="requires_repair", error=str(error))
        raise


if __name__ == "__main__":
    main()
