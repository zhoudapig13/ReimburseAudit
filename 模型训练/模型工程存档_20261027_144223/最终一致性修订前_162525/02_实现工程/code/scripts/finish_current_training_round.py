"""Finish the running training stage, evaluate once, then stop."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from audit_engine.training.continuous import validation_rank  # noqa: E402


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def alive(pid):
    path = Path(f"/proc/{pid}/stat")
    try:
        return path.read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    lock = (ROOT / "runs/continuous_official_training.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = folder / "state.json"
    state = json.loads(state_path.read_text())
    cycle = state["cycle"]
    if state["stage"] != f"cycle_{cycle:03d}_training":
        raise ValueError("expected an active training stage")
    trainer_pid = state["child_pid"]
    state.update(controller_pid=os.getpid(), stop_after_current_round=True)
    write_json(state_path, state)
    try:
        while alive(trainer_pid):
            state["updated_unix"] = time.time()
            write_json(state_path, state)
            time.sleep(5)
        trained = json.loads(
            (folder / f"cycle_{cycle:03d}_train/report.json").read_text()
        )
        if (
            trained["status"] != "passed"
            or trained["source_truncated"]
            or trained["deferred_workflows"]
        ):
            raise ValueError("training did not finish successfully")
        checkpoint = Path(trained["final_checkpoint"])
        output = folder / f"cycle_{cycle:03d}_validation"
        output.mkdir()
        command = [
            str(ROOT / ".venv/bin/python"),
            "-u",
            "code/scripts/evaluate_official_adapter.py",
            "--config",
            state["config"],
            "--adapter",
            str(checkpoint),
            "--device",
            "cuda:3",
            "--token-budget",
            "131072",
            "--output-dir",
            str(output),
            "--adapter-only",
        ]
        with (output / "execution.log").open("w") as log:
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
                stage=f"cycle_{cycle:03d}_validation",
                child_pid=child.pid,
                command=command,
            )
            write_json(state_path, state)
            while child.poll() is None:
                state["updated_unix"] = time.time()
                write_json(state_path, state)
                time.sleep(5)
            if child.returncode:
                raise RuntimeError(f"validation exited {child.returncode}")
        report = json.loads((output / "report.json").read_text())
        if (
            report["validation_evaluated"] != 23
            or report["validation_available"] != 23
            or report["deferred_workflow_ids"]
            or report["test_targets_loaded"]
            or report["source_truncated"]
        ):
            raise ValueError("validation coverage/isolation check failed")
        current = report["adapter"]
        if validation_rank(current) > validation_rank(state["best_metrics"]):
            state.update(best_checkpoint=str(checkpoint), best_metrics=current)
        summary = {
            "cycle": cycle,
            "completed_beijing": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "training_hours_this_round": trained["training_seconds_this_run"] / 3600,
            "training_hours_total": trained["training_seconds_total"] / 3600,
            "optimizer_steps": trained["optimizer_steps"],
            "validation_workflows": 23,
            "adapter": current,
            "base": state["base_metrics"],
            "best": state["best_metrics"],
            "checkpoint": str(checkpoint),
            "report": str(output / "report.json"),
            "action": "stop_by_user_after_evaluation",
            "explanation_metric": "exact text match, not semantic grading",
        }
        state["rounds"].append(summary)
        write_json(folder / f"result_{cycle:03d}.json", summary)
        write_json(folder / "latest_result.json", summary)
        state.update(
            status="stopped_by_user",
            stage="validation_complete",
            child_returncode=0,
            finished_unix=time.time(),
            updated_unix=time.time(),
        )
        write_json(state_path, state)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    except Exception as error:
        state.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            updated_unix=time.time(),
        )
        write_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
