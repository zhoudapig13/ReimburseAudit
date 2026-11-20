"""Check GPU availability once at a given time and resume official training."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def availability():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    gpus = []
    for line in result.stdout.splitlines():
        index, memory, utilization = (int(value.strip()) for value in line.split(","))
        gpus.append(
            {"index": index, "memory_mib": memory, "utilization_percent": utilization}
        )
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    ).stdout.strip()
    idle = (
        {gpu["index"] for gpu in gpus} == set(range(8))
        and all(
            gpu["memory_mib"] <= 512 and gpu["utilization_percent"] <= 5 for gpu in gpus
        )
        and not processes
    )
    return {"idle": idle, "gpus": gpus, "compute_pids": processes.splitlines()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--at", required=True, help="ISO datetime including timezone")
    parser.add_argument("--resume-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    due = datetime.fromisoformat(args.at)
    if due.tzinfo is None:
        parser.error("--at must include timezone")
    if args.check_only:
        print(json.dumps(availability(), ensure_ascii=False))
        return
    folder = args.output_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / "schedule.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = folder / "schedule.json"
    if status_path.exists():
        raise ValueError("schedule already exists")
    state = {
        "status": "waiting",
        "scheduled_at": due.isoformat(),
        "scheduler_pid": os.getpid(),
    }

    def save():
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(status_path)

    save()
    try:
        while (remaining := due.timestamp() - time.time()) > 0:
            time.sleep(min(remaining, 30))
        state["checked_unix"] = time.time()
        state["availability"] = availability()
        if not state["availability"]["idle"]:
            state["status"] = "skipped_busy"
            save()
            return
        command = [
            str(ROOT / ".venv/bin/python"),
            "-u",
            "code/scripts/run_continuous_official_training.py",
            "--output-dir",
            str(folder / "training"),
            "--resume-run",
            str(args.resume_run.resolve()),
            "--first-hours",
            "2",
            "--later-hours",
            "2",
        ]
        with (folder / "controller.log").open("a") as log:
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=dict(
                    os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
                ),
            )
        state.update(
            status="launched",
            controller_pid=child.pid,
            command=command,
            launched_unix=time.time(),
        )
        save()
        returncode = child.wait()
        state.update(
            status="completed" if returncode == 0 else "failed", returncode=returncode
        )
        save()
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
