"""Resume-safe actual-reader merge, frozen-feature caching and pilot training."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit_engine.contracts import AuditCase

ROOT = Path(__file__).resolve().parents[2]


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader-root", type=Path, action="append", required=True)
    parser.add_argument("--selection", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/patent_delivery.yaml"))
    parser.add_argument("--max-gpus", type=int, default=1)
    parser.add_argument("--wait-process-record", type=Path, action="append", default=[])
    args = parser.parse_args()
    if not 1 <= args.max_gpus <= 5:
        parser.error("max-gpus must be in 1..5")
    args.output.mkdir(parents=True, exist_ok=True)
    status = {
        "pid": os.getpid(),
        "stage": "waiting_for_actual_reader_cases",
        "automatic_accuracy_stop": False,
    }

    def save(**changes):
        status.update(changes, updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(args.output / "pipeline_status.json", status)

    expected = {
        identifier
        for selection in args.selection
        for identifier in json.loads(selection.read_text())["workflow_ids"]
    }
    selected = {}
    while True:
        selected, errors = {}, []
        for root in args.reader_root:
            for path in root.glob("shard_*/*.json"):
                if path.name == "selection.json":
                    continue
                record = json.loads(path.read_text())
                identifier = record.get("workflow_id")
                if identifier not in expected:
                    continue
                if record.get("status") == "failed":
                    errors.append(
                        {
                            "workflow_id": identifier,
                            "source": str(path),
                            "error": record.get("error"),
                        }
                    )
                    continue
                case = AuditCase.model_validate(record["case"])
                if case.digest() != record[
                    "case_sha256"
                ] or not case.input_version.startswith("patent_reader_v1:"):
                    raise ValueError(
                        "reader case is not bound to the recorded actual input"
                    )
                obtained = sum(o.status == "obtained" for o in case.operands)
                if obtained == 0:
                    errors.append(
                        {"workflow_id": identifier, "error": "no_actual_obtained_facts"}
                    )
                    continue
                if identifier in selected:
                    raise ValueError("duplicate workflow reader output")
                selected[identifier] = (
                    case,
                    path,
                    len(record.get("issues", [])),
                    obtained,
                )
        save(
            reader_cases=len(selected),
            expected_cases=len(expected),
            reader_failures=errors,
            missing_workflows=sorted(expected - selected.keys()),
        )
        if errors:
            save(stage="reader_input_requires_repair")
            return 2
        if set(selected) == expected:
            break
        time.sleep(10)
    cases = args.output / "reader_cases.jsonl"
    if not cases.exists():
        cases.write_text(
            "".join(selected[i][0].model_dump_json() + "\n" for i in sorted(expected)),
            encoding="utf-8",
        )
    elif cases.read_text() != "".join(
        selected[i][0].model_dump_json() + "\n" for i in sorted(expected)
    ):
        raise ValueError(
            "completed training inputs changed; use a new output directory"
        )
    atomic_json(
        args.output / "reader_quality.json",
        {
            i: {"record": str(v[1]), "issues": v[2], "obtained_facts": v[3]}
            for i, v in selected.items()
        },
    )
    annotations = args.output / "annotations.jsonl"

    def command(arguments, name, gpu=None):
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log = args.output / (name + ".log")
        with log.open("ab") as stream:
            child = subprocess.Popen(
                [sys.executable, "-u", *map(str, arguments)],
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return child, {"pid": child.pid, "gpu": gpu, "log": str(log)}

    child, record = command(
        [
            "code/scripts/compile_source_annotations.py",
            "--cases",
            cases,
            "--output",
            annotations,
        ],
        "compile_annotations",
    )
    save(stage="binding_source_supervision", job=record)
    if child.wait():
        save(stage="source_supervision_failed")
        return 3
    features = args.output / "features"
    for path in args.wait_process_record:
        records = json.loads(path.read_text())
        if isinstance(records, dict):
            records = [records]
        while True:
            running = []
            for record in records:
                proc = Path("/proc") / str(record["pid"]) / "stat"
                try:
                    if not proc.read_text().split(") ", 1)[1].startswith("Z"):
                        running.append(record["pid"])
                except FileNotFoundError:
                    pass
            if not running:
                break
            save(stage="waiting_for_reader_gpu_release", reader_pids=running)
            time.sleep(10)
    while True:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        available = [
            int(line.split(",")[0])
            for line in result.splitlines()
            if int(line.split(",")[1]) < 1024 and int(line.split(",")[2]) <= 10
        ]
        if available:
            break
        save(stage="waiting_for_idle_gpu")
        time.sleep(10)
    gpus = available[: args.max_gpus]
    jobs = [
        command(
            [
                "code/scripts/patent_pipeline.py",
                "encode",
                "--cases",
                cases,
                "--features",
                features,
                "--config",
                args.config,
                "--model-path",
                args.model_path,
                "--shard",
                f"{i}/{len(gpus)}",
            ],
            f"encode_{i}",
            gpu,
        )
        for i, gpu in enumerate(gpus)
    ]
    save(
        stage="encoding_actual_frozen_backbone_features",
        jobs=[record for _, record in jobs],
    )
    codes = [child.wait() for child, _ in jobs]
    if any(codes):
        save(stage="feature_encoding_failed", exit_codes=codes)
        return 4
    child, record = command(
        [
            "code/scripts/patent_pipeline.py",
            "train",
            "--cases",
            cases,
            "--annotations",
            annotations,
            "--features",
            features,
            "--config",
            args.config,
            "--output",
            args.output / "weights",
        ],
        "train",
        gpus[0],
    )
    save(
        stage="training_three_patent_components",
        training_started_at=datetime.now(timezone.utc).isoformat(),
        job=record,
    )
    if child.wait():
        save(stage="training_failed")
        return 5
    summary = json.loads((args.output / "weights/training_summary.json").read_text())
    save(
        stage="pilot_checkpoint_ready"
        if summary["three_components_optimized"]
        else "component_optimization_incomplete",
        three_components_optimized=summary["three_components_optimized"],
        checkpoint=str(args.output / "weights/audit_heads.pt"),
        nonzero_gradient_steps=summary["nonzero_gradient_steps"],
        final_delivery_closed=False,
        full_23_case_accuracy=None,
    )
    return 0 if summary["three_components_optimized"] else 6


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"pipeline failure: {type(error).__name__}: {error}", flush=True)
        raise
