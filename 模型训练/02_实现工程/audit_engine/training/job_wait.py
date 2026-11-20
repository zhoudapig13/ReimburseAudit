"""Wait for a verified training job, including process startup before state creation."""

from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "work/job_wait"


def read_json(path):
    return json.loads(path.read_text())


def write_state(state):
    CONTROL.mkdir(parents=True, exist_ok=True)
    state["updated_unix"] = time.time()
    temporary = CONTROL / "state.tmp"
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(CONTROL / "state.json")


def live_training(state, output):
    pid = state.get("pid")
    if not isinstance(pid, int):
        return False
    try:
        arguments = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return (
            Path(f"/proc/{pid}/cwd").resolve() == ROOT
            and any(b"train_reader_lora.py" in arg or b"patent_pipeline.py" in arg for arg in arguments)
            and str(output.relative_to(ROOT)).encode() in arguments
        )
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def wait_completed(directory, state):
    while True:
        if not (directory / "state.json").exists():
            candidates = [
                int(entry.name)
                for entry in Path("/proc").iterdir()
                if entry.name.isdecimal()
                and live_training({"pid": int(entry.name)}, directory)
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"no unique live prerequisite for missing state: {directory}"
                )
            state.update(
                waiting_for=str(directory.relative_to(ROOT)),
                verified_live_pid=candidates[0],
                prerequisite_stage="starting_before_state_file",
            )
            write_state(state)
            time.sleep(1)
            continue
        job = read_json(directory / "state.json")
        if job["status"] == "completed":
            report = read_json(directory / "report.json")
            if report["status"] != "passed" or report["optimizer_steps"] < 1:
                raise ValueError(f"job has no verified optimization: {directory}")
            return job, report
        if job["status"] == "failed":
            raise RuntimeError(f"prerequisite failed: {directory}: {job.get('error')}")
        if not live_training(job, directory):
            # Recheck completion after the process-exit/report-write race.
            time.sleep(1)
            job = read_json(directory / "state.json")
            if job["status"] != "completed":
                raise RuntimeError(f"prerequisite process is missing: {directory}")
            continue
        state.update(
            waiting_for=str(directory.relative_to(ROOT)),
            verified_live_pid=job["pid"],
            prerequisite_stage=job["status"],
        )
        write_state(state)
        time.sleep(10)


