"""Select a bounded fine-tuning trial using source labels, then score once."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import yaml

from audit_engine.contracts import AuditCase, ConditionAnnotation
from audit_engine.models.heads import AuditHeads
from audit_engine.training.patent import (component_validation, correction_validation, feature_path,
    load_dataset, load_features, read_jsonl, validate_annotations)


def main():
    root = Path("runs/patent_confirmed_gap_20261027_0947")
    previous = Path("runs/patent_targeted_source_20261027_0930")
    trials_root = Path("runs/patent_policy_balance_20261027_0943")
    state = {"pid": os.getpid(), "final_delivery_closed": False, "reference_answers_used_for_selection": False}
    torch.set_num_threads(4)

    def save(**values):
        state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        tmp = root / "status.tmp"
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        tmp.replace(root / "status.json")

    def start(arguments, name, gpu=None):
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with (root / (name + ".log")).open("ab") as log:
            return subprocess.Popen([sys.executable, "-u", *map(str, arguments)], env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def run(arguments, name):
        child = start(arguments, name)
        save(child_pid=child.pid, command_name=name)
        if child.wait():
            raise RuntimeError(name + " failed; inspect its saved log")

    try:
        readers = json.loads((root / "reader_jobs.json").read_text())
        trials = json.loads((trials_root / "processes.json").read_text())
        save(stage="waiting_for_targeted_reads_and_fine_tuning")
        while any(Path("/proc/" + str(j["pid"])).exists() for j in readers + trials):
            time.sleep(10)
        compiled = {c["workflow_id"]: AuditCase.model_validate(c) for c in read_jsonl(Path("data/patent_reviewed/policy_cases_v15.jsonl"))}
        changed = {}
        for path in Path("data/patent_confirmed_gap_reader_v15").glob("shard_*/*.json"):
            row = json.loads(path.read_text())
            if "case" not in row:
                continue
            case = AuditCase.model_validate(row["case"])
            if row["source_case_sha256"] != compiled[case.workflow_id].digest() or case.digest() != row["case_sha256"]:
                raise ValueError("changed source reading checksum differs")
            changed[case.workflow_id] = case
        if len(changed) != 4:
            raise ValueError("missing confirmed-gap reader results")
        dataset_path = root / "dataset"
        dataset_path.mkdir(exist_ok=True)
        cases = {row["workflow_id"]: AuditCase.model_validate(row) for row in read_jsonl(previous / "dataset/reader_cases.jsonl")}
        cases.update(changed)
        partitions = {row["workflow_id"]: row for row in read_jsonl(previous / "dataset/partitions.jsonl")}
        annotations = []
        for row in read_jsonl(previous / "dataset/annotations.jsonl"):
            row["case_sha256"] = cases[row["workflow_id"]].digest()
            annotations.append(ConditionAnnotation.model_validate(row))
        validate_annotations(list(cases.values()), annotations, partitions)
        inputs, labels = dataset_path / "reader_cases.jsonl", dataset_path / "annotations.jsonl"
        inputs.write_text("".join(case.model_dump_json() + "\n" for case in cases.values()))
        labels.write_text("".join(row.model_dump_json() + "\n" for row in annotations))
        shutil.copy2(previous / "dataset/partitions.jsonl", dataset_path / "partitions.jsonl")
        features = root / "features"
        features.mkdir(exist_ok=True)
        for case in cases.values():
            if case.workflow_id in changed:
                continue
            source = feature_path(previous / "features", case.workflow_id)
            load_features(source).validate(case, 4096)
            target = feature_path(features, case.workflow_id)
            if not target.exists():
                shutil.copy2(source, target)
        config = yaml.safe_load((previous / "config.yaml").read_text())
        config["data"]["partitions"] = str(dataset_path / "partitions.jsonl")
        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        changed_path = root / "changed_cases.jsonl"
        changed_path.write_text("".join(case.model_dump_json() + "\n" for case in changed.values()))
        save(stage="encoding_revised_conditions")
        children = [start(["code/scripts/patent_pipeline.py", "encode", "--cases", changed_path,
                          "--features", features, "--config", config_path,
                          "--model-path", "/mnt/public/zhouyibo/.cache/reimbursement-patent/model/Qwen3-VL-8B-Instruct",
                          "--shard", str(index) + "/4"], "encode_" + str(index), gpu)
                    for index, gpu in enumerate([1, 2, 3, 4])]
        codes = [child.wait() for child in children]
        if any(codes):
            raise RuntimeError("revised feature encoding failed: " + str(codes))
        save(stage="selecting_by_source_condition_validation")
        dataset = load_dataset(inputs, labels, features, dataset_path / "partitions.jsonl")
        candidates = [{"gain": 1.0, "output": str(previous / "weights")} ] + trials
        rankings = []
        for trial in candidates:
            checkpoint = Path(trial["output"]) / "audit_heads.pt"
            bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
            cfg = bundle["metadata"]["config"]
            c = cfg["model"]
            model = AuditHeads(c["hidden_size"], c["dimension"], c["width"], c["role_slots"])
            model.load_state_dict(bundle["model"], strict=True)
            metrics = component_validation(model, dataset["validation"], cfg)
            corrections = correction_validation(model, dataset["validation"], cfg)
            key = [-(metrics["state_macro_f1"] or 0.0), -(metrics["reference_recall"] or 0.0),
                   metrics["reference_false_positive_fraction"], -(corrections["workflow_mean_net_correction"] or 0.0),
                   corrections["unwanted_changes"]]
            rankings.append({"gain": trial["gain"], "checkpoint": str(checkpoint),
                             "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                             "component_validation": metrics, "correction_validation": corrections, "selection_key": key})
        rankings.sort(key=lambda row: row["selection_key"])
        selected = rankings[0]
        (root / "source_validation_selection.json").write_text(json.dumps({"criterion": "state_macro_f1_then_reference_recall_then_false_positive_fraction_then_correction_net", "benchmark_answers_read": False,
            "candidates": rankings, "selected": selected}, ensure_ascii=False, indent=2))
        validation_ids = [row["workflow_id"] for row in read_jsonl(previous / "validation_cases.jsonl")]
        validation = root / "validation_cases.jsonl"
        validation.write_text("".join(cases[key].model_dump_json() + "\n" for key in validation_ids))
        predictions = root / "predictions.jsonl"
        save(stage="evaluating_fixed_23_workflows", selected_checkpoint=selected["checkpoint"])
        run(["code/scripts/patent_pipeline.py", "predict", "--cases", validation, "--features", features,
             "--checkpoint", selected["checkpoint"], "--output", predictions, "--device", "cpu"], "predict")
        rows = list(read_jsonl(predictions))
        if len(rows) != 23 or {row["workflow_id"] for row in rows} != set(validation_ids) or any(row.get("status") == "failed" for row in rows):
            raise ValueError("full validation inference is incomplete")
        save(stage="scoring_frozen_predictions", predictions_sha256=hashlib.sha256(predictions.read_bytes()).hexdigest())
        run(["code/scripts/evaluate_patent_predictions.py", "--predictions", predictions,
             "--gold", "data/labels_private/validation_reference_targets.jsonl", "--output", root / "score.json"], "score")
        score = json.loads((root / "score.json").read_text())
        save(stage="evaluation_complete", correct=score["correct"], total=score["total"], accuracy=score["accuracy"],
             accuracy_target_met=score["accuracy"] > 0.5)
    except Exception as error:
        save(stage="requires_repair", error=str(error))
        raise


if __name__ == "__main__":
    main()
