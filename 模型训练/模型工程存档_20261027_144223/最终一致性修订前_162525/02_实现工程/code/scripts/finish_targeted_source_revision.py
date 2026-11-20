"""Train on additional source-reviewed TRAIN examples, then freeze/score all 23."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import yaml

from audit_engine.contracts import AuditCase, ConditionAnnotation
from audit_engine.training.patent import feature_path, load_features, read_jsonl, validate_annotations
from compile_source_annotations import compile_annotations


ROOT = Path("runs/patent_targeted_source_20261027_0930")
PREVIOUS = Path("runs/patent_shared_official_20261027_0822")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ROOT.mkdir(exist_ok=True)
    status = {"pid": os.getpid(), "final_delivery_closed": False,
              "held_out_reference_answers_used_for_training": False}

    def save(**values):
        status.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        temporary = ROOT / "status.tmp"
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2))
        temporary.replace(ROOT / "status.json")

    def start(arguments, name, gpu=None):
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with (ROOT / (name + ".log")).open("ab") as log:
            child = subprocess.Popen([sys.executable, "-u", *map(str, arguments)], env=env,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        return child

    def run(arguments, name):
        child = start(arguments, name)
        save(child_pid=child.pid, command_name=name)
        if child.wait():
            raise RuntimeError(name + " failed; inspect the recorded log")

    try:
        jobs = json.loads((ROOT / "reader_jobs_retry.json").read_text())
        save(stage="reading_targeted_materials", readers=jobs)
        while any(Path("/proc/" + str(job["pid"])).exists() for job in jobs):
            time.sleep(10)
        changed_ids = set(json.loads((ROOT / "selection.json").read_text())["workflow_ids"])
        compiled = {row["workflow_id"]: AuditCase.model_validate(row)
                    for row in read_jsonl(Path("data/patent_reviewed/policy_cases_v14.jsonl"))}
        changed = {}
        for directory in (Path("data/patent_targeted_reader_v14_retry_train"),
                          Path("data/patent_targeted_reader_v14_retry_validation"),
                          Path("data/patent_targeted_field_replay_v14")):
            for path in directory.glob("shard_*/*.json"):
                row = json.loads(path.read_text())
                if row.get("workflow_id") not in changed_ids or "case" not in row:
                    continue
                identifier = row["workflow_id"]
                if row["source_case_sha256"] != compiled[identifier].digest():
                    raise ValueError("targeted reader used stale policy inputs")
                case = AuditCase.model_validate(row["case"])
                if case.digest() != row["case_sha256"]:
                    raise ValueError("targeted source reading checksum changed")
                changed[identifier] = case
        if set(changed) != changed_ids:
            raise ValueError("missing actual readings: " + str(sorted(changed_ids - set(changed))))
        save(stage="preparing_source_supervision", actual_read_workflows=len(changed))
        dataset = ROOT / "dataset"
        dataset.mkdir(exist_ok=True)
        previous = {row["workflow_id"]: AuditCase.model_validate(row)
                    for row in read_jsonl(PREVIOUS / "dataset/reader_cases.jsonl")}
        previous.update(changed)
        cases = list(previous.values())
        partitions = {row["workflow_id"]: row for row in read_jsonl(PREVIOUS / "dataset/partitions.jsonl")}
        original_partitions = {row["workflow_id"]: row for row in read_jsonl(Path("data/patent_full/partitions.jsonl"))}
        partitions.update({key: original_partitions[key] for key in changed})
        annotations = [ConditionAnnotation.model_validate(row)
                       for row in read_jsonl(PREVIOUS / "dataset/annotations.jsonl")
                       if row["workflow_id"] not in changed]
        for key, case in changed.items():
            review = json.loads((Path("data/patent_source_review_v14") / (key + ".json")).read_text())
            if partitions[key]["split"] == "train":
                annotations.extend(compile_annotations(case, review, partitions[key]))
                continue
            # Source-level validation supervision, never benchmark answers.
            # Each target below comes from the separately inspected original
            # receipt, travel explanation, roster and policy, not model outputs.
            source_states = {
                "shared_meal_grade_cap": ("satisfied", ["e003", "e005", "e006"]),
                "formal_computer_generated_receipt": ("satisfied", ["e006"]),
                "required_receipt_fields_present": ("failed" if key == "1.2.4" else "satisfied", ["e006"]),
                "meal_date_matches_documented_delay": ("insufficient" if key == "1.2.4" else "satisfied", ["e001", "e006"]),
                "shared_meal_participant_count": ("failed" if key == "1.2.1" else "satisfied", ["e003", "e006"]),
                "expense_is_a_meal": ("failed" if key == "1.2.2" else "satisfied", ["e006"]),
                "merchant_city_matches_travel": ("satisfied", ["e001", "e006"]),
                "documented_force_majeure_reason": ("satisfied", ["e001"]),
            }
            inspected = json.loads(json.dumps(review))
            for rule in inspected["confirmed_policy"]:
                state, providers = source_states[rule["id"]]
                rule.update(candidate_state=state, positive_sources=[], conflict_sources=[], negative_sources=[])
                # An undated original receipt supplies no date relation.
                if state == "insufficient":
                    continue
                rule["positive_sources" if state == "satisfied" else "conflict_sources"] = providers
                rule["negative_sources"] = sorted(set(inspected["source_sha256"]) - set(providers))
            (ROOT / (key + "_source_validation_review.json")).write_text(json.dumps(inspected, ensure_ascii=False, indent=2))
            annotations.extend(compile_annotations(case, inspected, partitions[key]))
        validate_annotations(cases, annotations, partitions)
        inputs = dataset / "reader_cases.jsonl"
        labels = dataset / "annotations.jsonl"
        splits = dataset / "partitions.jsonl"
        inputs.write_text("".join(case.model_dump_json() + "\n" for case in cases))
        labels.write_text("".join(row.model_dump_json() + "\n" for row in annotations))
        splits.write_text("".join(json.dumps(row) + "\n" for row in partitions.values()))
        features = ROOT / "features"
        features.mkdir(exist_ok=True)
        for case in cases:
            if case.workflow_id in changed:
                continue
            source = feature_path(PREVIOUS / "features", case.workflow_id)
            feature = load_features(source)
            feature.validate(case, 4096)
            target = feature_path(features, case.workflow_id)
            if not target.exists():
                shutil.copy2(source, target)
        encoding_inputs = ROOT / "changed_cases.jsonl"
        encoding_inputs.write_text("".join(case.model_dump_json() + "\n" for case in changed.values()))
        config = yaml.safe_load((PREVIOUS / "config.yaml").read_text())
        config["data"]["partitions"] = str(splits)
        config["heads"].update(learning_rate=5e-5, policy_learning_rate_multiplier=6.0)
        config["correction"]["learning_rate"] = 3e-5
        config["fast_training"] = {"epochs": {"B1": 12, "B2": 6, "B3": 10, "B4": 4, "C": 6}, "patience": 3}
        configuration = ROOT / "config.yaml"
        configuration.write_text(yaml.safe_dump(config, sort_keys=False))
        save(stage="encoding_changed_materials", unchanged_feature_caches_reused=len(cases)-len(changed), changed_workflows=len(changed))
        children = [start(["code/scripts/patent_pipeline.py", "encode", "--cases", encoding_inputs,
                           "--features", features, "--config", configuration,
                           "--model-path", "/mnt/public/zhouyibo/.cache/reimbursement-patent/model/Qwen3-VL-8B-Instruct",
                           "--shard", str(index) + "/5"], "encode_" + str(index), gpu)
                    for index, gpu in enumerate([1, 2, 3, 4, 5])]
        save(encoding_pids=[child.pid for child in children])
        codes = [child.wait() for child in children]
        if any(codes):
            raise RuntimeError("targeted encoding failed: " + str(codes))
        save(stage="fine_tuning_three_components")
        run(["code/scripts/patent_pipeline.py", "train", "--cases", inputs, "--annotations", labels,
             "--features", features, "--config", configuration, "--initial-checkpoint", PREVIOUS / "weights/audit_heads.pt",
             "--device", "cpu", "--output", ROOT / "weights"], "train")
        summary = json.loads((ROOT / "weights/training_summary.json").read_text())
        if not summary["three_components_optimized"] or any(summary["stage_steps"].get(stage, 0) <= 0 for stage in ("B1", "B2", "B3", "B4", "C")):
            raise ValueError("not all required training stages actually ran")
        validation_ids = [row["workflow_id"] for row in read_jsonl(PREVIOUS / "validation_cases.jsonl")]
        validation = ROOT / "validation_cases.jsonl"
        validation.write_text("".join(previous[key].model_dump_json() + "\n" for key in validation_ids))
        save(stage="evaluating_fixed_23_workflows")
        predictions = ROOT / "predictions.jsonl"
        run(["code/scripts/patent_pipeline.py", "predict", "--cases", validation, "--features", features,
             "--checkpoint", ROOT / "weights/audit_heads.pt", "--output", predictions, "--device", "cpu"], "predict")
        saved = list(read_jsonl(predictions))
        if len(saved) != 23 or {row["workflow_id"] for row in saved} != set(validation_ids) or any(row.get("status") == "failed" for row in saved):
            raise ValueError("full 23-workflow inference did not complete successfully")
        save(stage="scoring_frozen_predictions", predictions_sha256=digest(predictions), checkpoint_sha256=digest(ROOT / "weights/audit_heads.pt"))
        run(["code/scripts/evaluate_patent_predictions.py", "--predictions", predictions,
             "--gold", "data/labels_private/validation_reference_targets.jsonl", "--output", ROOT / "score.json"], "score")
        score = json.loads((ROOT / "score.json").read_text())
        save(stage="evaluation_complete", correct=score["correct"], total=score["total"], accuracy=score["accuracy"],
             accuracy_target_met=score["accuracy"] > 0.5)
    except Exception as error:
        save(stage="requires_repair", error=str(error))
        raise


if __name__ == "__main__":
    main()
