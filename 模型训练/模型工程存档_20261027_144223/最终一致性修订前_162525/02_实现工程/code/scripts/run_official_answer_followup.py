"""After the live 95-workflow run, train the ten complete deferred workflows."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "runs/official_sft_long_train"
PROBE = ROOT / "runs/official_sft_registry_chunked_probe"
STABILITY = ROOT / "runs/official_sft_maxlength_stability"
PREFLIGHT = ROOT / "runs/official_sft_complete_preflight"
OUTPUT = ROOT / "runs/official_sft_supplement"
CONTROL = ROOT / "runs/official_sft_followup"


def read_json(path):
    return json.loads(path.read_text())


def write_state(state):
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
            and any(b"train_official_answers.py" in arg for arg in arguments)
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


def main():
    CONTROL.mkdir(parents=True, exist_ok=True)
    if (CONTROL / "state.json").exists():
        raise ValueError(
            "followup controller already has a state; inspect before restarting"
        )
    state = {
        "controller_pid": os.getpid(),
        "status": "waiting",
        "started_unix": time.time(),
    }
    write_state(state)
    try:
        preflight = read_json(PREFLIGHT / "stage_manifest.json")
        if preflight["selected_workflows"] != 105 or preflight["deferred_workflows"]:
            raise ValueError(
                "complete source preflight did not cover all 105 training workflows"
            )
        wait_completed(PROBE, state)
        _, stability_report = wait_completed(STABILITY, state)
        if stability_report["optimizer_steps"] < 20:
            raise ValueError(
                "maximum-length stability check has fewer than twenty steps"
            )
        parent_state, parent_report = wait_completed(PARENT, state)
        rows = [json.loads(line) for line in (PARENT / "dataset_manifest.jsonl").open()]
        selected = {
            r["workflow_id"]
            for r in rows
            if r["status"] == "selected_complete_workflow"
        }
        remaining = sorted(
            r["workflow_id"]
            for r in rows
            if r["status"] != "selected_complete_workflow"
        )
        if len(selected) != 95 or len(remaining) != 10 or selected & set(remaining):
            raise ValueError("95+10 unique workflow coverage mismatch")
        for path, digest in parent_report["source_sha256"].items():
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
                raise ValueError("training inputs changed after the parent run")
        checkpoint = Path(parent_state["checkpoint"])
        if (
            checkpoint != Path("runs/official_sft_long_train/checkpoint_epoch_3")
            or not (ROOT / checkpoint / "adapter_model.safetensors").is_file()
        ):
            raise ValueError("parent final checkpoint is missing or unexpected")
        if OUTPUT.exists():
            raise ValueError("supplement output already exists; refuse to overwrite")
        OUTPUT.mkdir()
        arguments = [
            str(ROOT / ".venv/bin/python"),
            "-u",
            str(ROOT / "code/scripts/train_official_answers.py"),
            "--config",
            "configs/official_sft5090_compact5gpu_chunked.yaml",
            "--output-dir",
            str(OUTPUT.relative_to(ROOT)),
            "--initial-adapter",
            str(checkpoint),
            "--device",
            "cuda:3",
        ]
        for workflow in remaining:
            arguments.extend(["--workflow-id", workflow])
        environment = dict(
            os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
        )
        with (OUTPUT / "training.log").open("w") as log:
            child = subprocess.Popen(
                arguments,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
            state.update(
                status="supplement_running",
                child_pid=child.pid,
                supplemental_workflows=remaining,
                parent_checkpoint=str(checkpoint),
                command=arguments,
            )
            write_state(state)
            while child.poll() is None:
                write_state(state)
                time.sleep(10)
            if child.returncode:
                raise RuntimeError(
                    f"supplement training exited {child.returncode}; see training.log"
                )
        report = read_json(OUTPUT / "report.json")
        if report["status"] != "passed" or report["selected_workflows"] != 10:
            raise ValueError(
                "supplement report did not prove ten workflow optimization"
            )
        state.update(
            status="completed",
            final_checkpoint="runs/official_sft_supplement/checkpoint_epoch_3",
            unique_training_workflows=105,
            heldout_accuracy_evaluated=False,
        )
        write_state(state)
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_state(state)
        raise


if __name__ == "__main__":
    main()
