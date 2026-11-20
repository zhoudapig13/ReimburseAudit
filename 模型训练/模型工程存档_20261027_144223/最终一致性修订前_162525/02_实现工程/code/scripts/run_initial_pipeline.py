"""Persistent controller for the initial Qwen/audit-head/pilot feature run."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    state_file = root / "runs/bootstrap/controller_state.json"
    stage = "starting"
    children = []

    def record(status, **extra):
        payload = {
            "controller_pid": os.getpid(),
            "stage": stage,
            "status": status,
            "child_pids": children,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **extra,
        }
        temporary = state_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(state_file)
        print(json.dumps(payload), flush=True)

    def execute(command, log_name):
        nonlocal children
        with (root / "runs/bootstrap" / log_name).open("a") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", *command], stdout=log, stderr=subprocess.STDOUT
            )
            children = [process.pid]
            record("running", command=command, log=log_name)
            result = process.wait()
        children = []
        if result != 0:
            raise RuntimeError(
                f"{stage} process exited with code {result}; see {log_name}"
            )

    try:
        stage = "checkpoint_download"
        # Resume cached ranges after a transient connection failure. Keep
        # retries bounded; model/schema/training failures still stop for repair.
        for attempt in range(3):
            try:
                execute(
                    ["code/scripts/download_checkpoint.py", "--workers", "24"],
                    "checkpoint_download.log",
                )
                break
            except RuntimeError:
                if attempt == 2:
                    raise
                record(
                    "retrying", completed_attempts=attempt + 1, retry_delay_seconds=30
                )
                time.sleep(30)
        stage = "real_backbone_and_head_smoke"
        execute(
            ["code/scripts/initial_training.py", "--verified-checkpoint"],
            "initial_training.log",
        )
        stage = "pilot_feature_encoding"
        processes = []
        logs = []
        for index, device in enumerate(("cuda:1", "cuda:2")):
            log = (root / f"runs/bootstrap/pilot_encode_{index}.log").open("a")
            logs.append(log)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "code/scripts/encode_pilot_features.py",
                    "--shard",
                    f"{index}/2",
                    "--device",
                    device,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            processes.append(process)
        children = [process.pid for process in processes]
        record("running")
        results = [process.wait() for process in processes]
        for log in logs:
            log.close()
        children = []
        if any(results):
            raise RuntimeError(f"pilot feature shards exited with codes {results}")
        stage = "pilot_feature_merge"
        execute(["code/scripts/merge_pilot_features.py"], "pilot_feature_merge.log")
        stage = "initial_pipeline_passed"
        record("passed", formal_training_started=False, approved_component_labels=0)
    except Exception as error:
        record("failed", error_type=type(error).__name__, message=str(error))
        raise


if __name__ == "__main__":
    main()
