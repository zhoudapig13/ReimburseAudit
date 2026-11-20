"""Freeze the current system and evaluate every fixed validation workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml

from audit_engine.contracts import AuditCase


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def merge_reader_case(compiled, record):
    """Accept actual readings only for exactly the reviewed policy version."""
    case = AuditCase.model_validate(record["case"])
    if case.digest() != record["case_sha256"]:
        raise ValueError("actual reader case digest changed")
    if record.get("source_case_sha256") != compiled.digest():
        raise ValueError("actual readings belong to a different compiled input version")
    if case.application != compiled.application or case.conditions != compiled.conditions:
        raise ValueError("reader changed application or compiled policies")
    if case.policy_numbers != compiled.policy_numbers or case.amount_programs != compiled.amount_programs:
        raise ValueError("reader changed source amount programs")
    if case.policy_coverage != compiled.policy_coverage:
        raise ValueError("reader changed policy coverage")
    original = [e.model_dump(exclude={"read_status"}) for e in compiled.evidence]
    observed = [e.model_dump(exclude={"read_status"}) for e in case.evidence]
    if original != observed:
        raise ValueError("reader changed source material content or ordering")
    return case


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--gpus", default="1,2,3,4,5")
    p.add_argument("--cases", type=Path, required=True,
                   help="Explicit reviewed policy inputs for this evaluation")
    p.add_argument("--reader-root", type=Path, required=True,
                   help="Actual source readings generated from exactly --cases")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    status = {"pid": os.getpid(), "stage": "freezing_current_inputs", "split": "validation",
              "official_answers_used_for_inference": False, "failed_cases_in_denominator": True}

    def save(**values):
        status.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        write_json(args.output / "status.json", status)

    save()
    shutil.copyfile(args.checkpoint, args.output / "checkpoint.pt")
    shutil.copyfile(args.config, args.output / "config.yaml")
    partitions = [json.loads(line) for line in Path("data/patent_full/partitions.jsonl").read_text().splitlines()]
    expected = [row["workflow_id"] for row in partitions if row["split"] == "validation"]
    if len(expected) != 23 or len(set(expected)) != 23:
        raise ValueError("expected the original 23-workflow validation partition")
    cases = {row["workflow_id"]: AuditCase.model_validate(row)
             for line in args.cases.read_text().splitlines()
             if (row := json.loads(line))["workflow_id"] in expected}
    actual_reads = []
    for source in args.reader_root.glob("shard_*/*.json"):
        if source.name == "selection.json":
            continue
        row = json.loads(source.read_text())
        identifier = row.get("workflow_id")
        if identifier not in cases or row.get("status") == "failed":
            continue
        cases[identifier] = merge_reader_case(cases[identifier], row)
        actual_reads.append(identifier)
    if set(cases) != set(expected):
        raise ValueError("incomplete validation inputs")
    inputs = args.output / "validation_cases.jsonl"
    inputs.write_text("".join(cases[key].model_dump_json() + "\n" for key in expected))
    coverage = {"total_workflows": len(expected), "actual_reader_workflows": sorted(actual_reads),
                "workflows_with_compiled_conditions": [key for key in expected if cases[key].conditions],
                "missing_compiled_conditions": [key for key in expected if not cases[key].conditions],
                "policy_coverage_verified": [key for key in expected if cases[key].policy_coverage == "verified"],
                "source_bound_amount_programs": [key for key in expected if cases[key].amount_programs],
                "evaluation_scope": "current_system_as_implemented_including_unresolved_inputs",
                "missing_data_repaired_using_reference_answers": False}
    write_json(args.output / "input_coverage.json", coverage)
    save(stage="encoding_all_23_workflows", expected_workflows=expected,
         checkpoint_sha256=digest(args.output / "checkpoint.pt"),
         cases_sha256=digest(inputs), config_sha256=digest(args.output / "config.yaml"),
         input_coverage=coverage)

    def start(command, name, gpu=None):
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log = args.output / (name + ".log")
        with log.open("ab") as stream:
            child = subprocess.Popen([sys.executable, "-u", *map(str, command)], env=env,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        return child, {"pid": child.pid, "gpu": gpu, "log": str(log)}

    gpus = [int(value) for value in args.gpus.split(",")]
    if not 1 <= len(gpus) <= 5 or len(set(gpus)) != len(gpus) or 6 in gpus:
        raise ValueError("use up to five distinct available GPUs, excluding faulty GPU 6")
    children = [start(["code/scripts/patent_pipeline.py", "encode", "--cases", inputs,
                      "--features", args.output / "features", "--config", args.output / "config.yaml",
                      "--model-path", args.model_path, "--shard", f"{index}/{len(gpus)}"],
                     f"encode_{index}", gpu) for index, gpu in enumerate(gpus)]
    save(encoding_jobs=[row for _, row in children])
    codes = [child.wait() for child, _ in children]
    save(stage="predicting_all_23_workflows", encoding_exit_codes=codes)
    predictions = args.output / "predictions.jsonl"
    child, job = start(["code/scripts/patent_pipeline.py", "predict", "--cases", inputs,
                        "--features", args.output / "features", "--checkpoint", args.output / "checkpoint.pt",
                        "--output", predictions, "--device", "cpu"], "predict")
    save(prediction_job=job)
    if child.wait():
        save(stage="prediction_requires_repair")
        return 2
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    if len(rows) != 23 or {row["workflow_id"] for row in rows} != set(expected):
        raise ValueError("predictions do not cover exactly the fixed 23 workflows")
    # Only the separate scorer opens the held-out answer file, after predictions
    # have been saved and their coverage checked.
    save(stage="scoring_saved_predictions", predictions_sha256=digest(predictions))
    child, job = start(["code/scripts/evaluate_patent_predictions.py", "--predictions", predictions,
                        "--gold", "data/labels_private/validation_reference_targets.jsonl",
                        "--output", args.output / "score.json"], "score")
    if child.wait():
        save(stage="scoring_requires_repair", scoring_job=job)
        return 3
    score = json.loads((args.output / "score.json").read_text())
    calls = {key: sum(row.get("module_calls", {}).get(key, 0) for row in rows)
             for key in ("relation", "contribution", "corrector")}
    save(stage="completed", total=score["total"], correct=score["correct"], accuracy=score["accuracy"],
         module_calls=calls, failed_predictions=sum(row.get("status") == "failed" for row in rows),
         missing_policy_workflows=len(coverage["missing_compiled_conditions"]),
         final_delivery_closed=False)
    print(json.dumps(status, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
