"""Detach-safe checkpoint recovery and launch of source-only pilot reader jobs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def alive(pid):
    path = Path("/proc") / str(pid) / "stat"
    try:
        return not path.read_text().split(") ", 1)[1].startswith("Z")
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/patent_full_20261026")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/patent_reader_pilot"
    )
    parser.add_argument("--max-gpus", type=int, default=5)
    parser.add_argument("--deadline", default="2026-10-27T08:00:00+08:00")
    args = parser.parse_args()
    if not 1 <= args.max_gpus <= 5:
        parser.error("max-gpus must be between 1 and 5")
    args.run.mkdir(parents=True, exist_ok=True)
    status_path = args.run / "reader_queue_status.json"
    state = {
        "pid": os.getpid(),
        "deadline": args.deadline,
        "stage": "waiting_for_base",
        "training_started": False,
    }

    def save():
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        state["deadline_reached"] = datetime.now(
            timezone.utc
        ) >= datetime.fromisoformat(args.deadline)
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        temporary.replace(status_path)

    env = os.environ.copy()
    cache = Path("/mnt/public/zhouyibo/.cache/reimbursement-patent")
    for key, relative in {
        "TMPDIR": "tmp",
        "HF_HOME": "huggingface",
        "PYTHONPYCACHEPREFIX": "pycache",
        "TORCHINDUCTOR_CACHE_DIR": "runtime/inductor",
        "TRITON_CACHE_DIR": "runtime/triton",
        "CUDA_CACHE_PATH": "runtime/cuda",
        "XDG_CACHE_HOME": "runtime/cache",
    }.items():
        target = cache / relative
        target.mkdir(parents=True, exist_ok=True)
        env[key] = str(target)
    env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    recovery = 0
    while True:
        save()
        checkpoint_record = args.run / "base_download.json"
        if checkpoint_record.is_file():
            record = json.loads(checkpoint_record.read_text())
            if (
                record.get("all_shard_sha256_verified") is True
                and record.get("revision") == "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
                and Path(record["local_path"]).resolve() == args.model_path.resolve()
            ):
                break
            raise ValueError(
                "checkpoint verification record does not match the pilot base"
            )
        process_record = args.run / "base_download_process.json"
        process = (
            json.loads(process_record.read_text()) if process_record.is_file() else {}
        )
        if not alive(process.get("pid", -1)):
            if recovery >= 2:
                state.update(
                    stage="base_download_failed", reason="recovery_budget_exhausted"
                )
                save()
                return 2
            recovery += 1
            log = args.run / f"base_download_recovery_{recovery}.log"
            with log.open("ab") as stream:
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "code/scripts/download_checkpoint.py",
                        "--config",
                        "configs/base512.yaml",
                        "--output-dir",
                        str(args.model_path),
                        "--record-path",
                        str(checkpoint_record),
                        "--root-files-only",
                        "--workers",
                        "8",
                        "--transfer-retries",
                        "8",
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            process_record.write_text(
                json.dumps({"pid": child.pid, "log": str(log), "workers": 8})
            )
            state["download_recovery_attempt"] = recovery
        time.sleep(10)
    state["stage"] = "waiting_for_idle_gpus"
    while True:
        save()
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        available = []
        for line in output.splitlines():
            index, used, total, utilization = map(int, line.split(","))
            if total - used >= 28000 and utilization <= 10:
                available.append(index)
        if available:
            break
        time.sleep(15)
    gpus = available[: args.max_gpus]
    args.output.mkdir(parents=True, exist_ok=True)
    children = []
    for index, gpu in enumerate(gpus):
        log = args.run / f"pilot_propose_shard_{index}.log"
        child_env = {**env, "CUDA_VISIBLE_DEVICES": str(gpu)}
        command = [
            sys.executable,
            "-u",
            "code/scripts/read_patent_cases.py",
            "--mode",
            "propose",
            "--model-path",
            str(args.model_path),
            "--output",
            str(args.output),
            "--shard",
            f"{index}/{len(gpus)}",
            "--device",
            "cuda:0",
            "--pilot-count",
            "20",
        ]
        pilot_list = ROOT / "data/patent_source_review/pilot_selection.json"
        if pilot_list.is_file():
            command.extend(["--workflow-list", str(pilot_list)])
        with log.open("ab") as stream:
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                env=child_env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        children.append(
            {
                "pid": child.pid,
                "gpu": gpu,
                "shard": f"{index}/{len(gpus)}",
                "log": str(log),
            }
        )
    state.update(stage="source_policy_proposals_running", jobs=children)
    while any(alive(child["pid"]) for child in children):
        save()
        time.sleep(15)
    records = [
        p for p in args.output.glob("shard_*/*.json") if p.name != "selection.json"
    ]
    summaries = [json.loads(p.read_text()) for p in records]
    state.update(
        stage="source_policy_proposals_ready_for_review"
        if len(summaries) == 20
        else "source_reader_jobs_incomplete",
        completed_workflows=len(summaries),
        failed_workflows=sum(r.get("status") == "failed" for r in summaries),
        proposed_conditions=sum(
            len(r.get("case", {}).get("conditions", [])) for r in summaries
        ),
    )
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
