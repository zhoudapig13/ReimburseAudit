"""Train from an existing checkpoint, evaluate the fixed validation set, audit calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-run", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--adapt-reader-encoder", action="store_true")
    args = parser.parse_args()
    source, run = args.dataset_run.resolve(), args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    state_path = run / "status.json"
    if state_path.exists():
        raise ValueError("cycle output already has a status; inspect before restarting")
    state = {"pid": os.getpid(), "status": "running", "jobs": [],
             "initial_checkpoint": str(args.initial_checkpoint.resolve()),
             "initial_checkpoint_sha256": hashlib.sha256(args.initial_checkpoint.read_bytes()).hexdigest(),
             "resume_mode": "warm_start_parameters_new_optimizers",
             "test_targets_loaded": False}

    def save(**values):
        state.update(values, updated_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(state_path)

    def execute(stage, arguments):
        command = [sys.executable, "-u", *map(str, arguments)]
        with (run / (stage + ".log")).open("x") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            state["jobs"].append({"stage": stage, "pid": child.pid, "command": command})
            save(stage=stage, child_pid=child.pid)
            code = child.wait()
        state["jobs"][-1]["exit_code"] = code
        save()
        if code:
            raise RuntimeError(f"{stage} exited {code}; inspect {stage}.log")

    try:
        execute("train", ["code/scripts/patent_pipeline.py", "train",
                "--cases", source / "dataset/reader_cases.jsonl",
                "--annotations", source / "dataset/annotations.jsonl",
                "--features", source / "features", "--config", run / "config.yaml",
                "--initial-checkpoint", args.initial_checkpoint,
                "--device", args.device, "--output", run / "weights",
                *(["--adapt-reader-encoder"] if args.adapt_reader_encoder else [])])
        summary = json.loads((run / "weights/training_summary.json").read_text())
        if not summary["three_components_optimized"] or any(
            summary["stage_steps"].get(stage, 0) <= 0 for stage in ("B1", "B2", "B3", "B4", "C")
        ):
            raise ValueError("not all mandatory training stages updated parameters")
        save(stage_steps=summary["stage_steps"], nonzero_gradient_steps=summary["nonzero_gradient_steps"])
        execute("predict", ["code/scripts/patent_pipeline.py", "predict",
                "--cases", source / "validation_cases.jsonl", "--features", source / "features",
                "--checkpoint", run / "weights/audit_heads.pt",
                "--output", run / "predictions.jsonl", "--device", args.device])
        predictions = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()]
        expected = {json.loads(line)["workflow_id"] for line in
                    (source / "validation_cases.jsonl").read_text().splitlines()}
        if len(predictions) != 23 or {p["workflow_id"] for p in predictions} != expected:
            raise ValueError("validation coverage differs from the fixed 23 workflows")
        if any(p.get("status") == "failed" for p in predictions):
            raise ValueError("validation contains a failed prediction")
        calls = {name: sum(p["module_calls"][name] for p in predictions)
                 for name in ("relation", "contribution", "corrector")}
        if not all(calls.values()) or any(p["ablation"] != "full" for p in predictions):
            raise ValueError("full three-module evaluation was not exercised")
        if any(not p["B"]["conservation_valid"] for p in predictions):
            raise ValueError("amount conservation failed")
        save(module_calls=calls, predictions_sha256=hashlib.sha256(
            (run / "predictions.jsonl").read_bytes()).hexdigest())
        execute("score", ["code/scripts/evaluate_patent_predictions.py",
                "--predictions", run / "predictions.jsonl",
                "--gold", "data/labels_private/validation_reference_targets.jsonl",
                "--output", run / "score.json"])
        result = json.loads((run / "score.json").read_text())
        save(status="complete", stage="evaluation_complete", correct=result["correct"],
             total=result["total"], accuracy=result["accuracy"],
             accuracy_target_met=result["accuracy"] > 0.5,
             checkpoint=str(run / "weights/audit_heads.pt"))
        ledger = run.parent / "evaluation_history.jsonl"
        with ledger.open("a") as stream:
            stream.write(json.dumps({k: state[k] for k in (
                "updated_at", "checkpoint", "correct", "total", "accuracy", "module_calls"
            )}, ensure_ascii=False) + "\n")
    except BaseException as error:
        save(status="failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
