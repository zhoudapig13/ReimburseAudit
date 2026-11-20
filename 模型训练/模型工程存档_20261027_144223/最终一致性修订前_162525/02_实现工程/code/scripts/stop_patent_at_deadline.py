"""Stop only this recorded patent run at the user's explicit deadline."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path


ALLOWED = {"resume_resource_queue.py", "run_reviewed_pilot.py", "read_patent_cases.py",
           "patent_pipeline.py", "train_heads_until.py", "compile_source_annotations.py"}


def recorded_pids(value):
    if isinstance(value, list):
        for item in value:
            yield from recorded_pids(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in {"pid", "pipeline_pid", "queue_pid"} and isinstance(item, int):
                yield item
            elif key in {"reader_pids", "running_pids"} and isinstance(item, list):
                yield from (pid for pid in item if isinstance(pid, int))
            elif isinstance(item, (dict, list)):
                yield from recorded_pids(item)


def owned(pid, root):
    try:
        arguments = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        names = {Path(arg.decode(errors="replace")).name for arg in arguments}
        return bool(names & ALLOWED) and Path(f"/proc/{pid}/cwd").resolve() == root
    except (OSError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--deadline", required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        parser.error("deadline must include an offset")
    run, root = args.run.resolve(), Path.cwd().resolve()
    record = {"pid": os.getpid(), "deadline": args.deadline, "stage": "armed",
              "timezone": "Asia/Shanghai", "final_delivery_closed": False}
    path = run / "deadline_status.json"

    def save(**values):
        record.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)

    save()
    while time.time() < deadline.timestamp():
        time.sleep(min(10, max(0.01, deadline.timestamp() - time.time())))
    # Stop launchers first; then gather their children from persisted job records.
    signalled = set()
    for _ in range(3):
        pids = set()
        for source in run.rglob("*.json"):
            try:
                pids.update(recorded_pids(json.loads(source.read_text())))
            except (OSError, ValueError):
                continue
        for pid in sorted(pids):
            if pid == os.getpid() or pid in signalled or not owned(pid, root):
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                signalled.add(pid)
            except ProcessLookupError:
                pass
        save(stage="stopping_recorded_run", signalled_pids=sorted(signalled))
        time.sleep(1)
    # The continuous trainer saves at the next optimizer boundary on SIGTERM.
    until = time.time() + 60
    while any(owned(pid, root) for pid in signalled) and time.time() < until:
        time.sleep(1)
    remaining = [pid for pid in signalled if owned(pid, root)]
    save(stage="deadline_stop_completed" if not remaining else "checkpoint_save_still_running",
         remaining_pids=remaining, training_stop_reason="user_requested_07_00")


if __name__ == "__main__":
    main()
