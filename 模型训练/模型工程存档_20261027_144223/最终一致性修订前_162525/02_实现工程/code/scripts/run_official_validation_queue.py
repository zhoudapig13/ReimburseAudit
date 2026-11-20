"""Run full validation automatically after the 105-workflow training chain."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEPENDENCY = ROOT / "runs/official_sft_followup/state.json"
CONTROL = ROOT / "runs/official_validation_queue"
OUTPUT = ROOT / "runs/official_sft_full_validation"


def write_state(state):
    state["updated_unix"] = time.time()
    temporary = CONTROL / "state.tmp"
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(CONTROL / "state.json")


def controller_live(pid):
    try:
        arguments = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return Path(f"/proc/{pid}/cwd").resolve() == ROOT and any(
            b"run_official_answer_followup.py" in a for a in arguments
        )
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def main():
    CONTROL.mkdir(parents=True, exist_ok=True)
    if (CONTROL / "state.json").exists():
        raise ValueError(
            "validation queue has an existing state; inspect it before restarting"
        )
    state = {
        "controller_pid": os.getpid(),
        "status": "waiting_for_training",
        "started_unix": time.time(),
    }
    write_state(state)
    try:
        while True:
            dependency = json.loads(DEPENDENCY.read_text())
            if dependency["status"] == "completed":
                break
            if dependency["status"] == "failed":
                raise RuntimeError(f"training chain failed: {dependency.get('error')}")
            if not controller_live(dependency["controller_pid"]):
                time.sleep(1)
                if json.loads(DEPENDENCY.read_text())["status"] == "completed":
                    continue
                raise RuntimeError("training-chain controller is missing")
            state.update(
                verified_training_controller_pid=dependency["controller_pid"],
                training_chain_stage=dependency["status"],
            )
            write_state(state)
            time.sleep(10)
        if dependency["unique_training_workflows"] != 105:
            raise ValueError("training chain did not prove 105-workflow coverage")
        checkpoint = dependency["final_checkpoint"]
        if (
            checkpoint != "runs/official_sft_supplement/checkpoint_epoch_3"
            or not (ROOT / checkpoint / "adapter_model.safetensors").is_file()
        ):
            raise ValueError("final adapter is missing or unexpected")
        training_bundles = [
            json.loads(line)
            for line in (ROOT / "data/normalized_full/train_inputs.jsonl").open()
        ]
        validation_bundles = [
            json.loads(line)
            for line in (ROOT / "data/normalized_full/validation_inputs.jsonl").open()
        ]
        coverage = set()
        for folder in ("official_sft_long_train", "official_sft_supplement"):
            rows = [
                json.loads(line)
                for line in (ROOT / f"runs/{folder}/dataset_manifest.jsonl").open()
            ]
            chosen = {
                row["workflow_id"]
                for row in rows
                if row["status"] == "selected_complete_workflow"
            }
            if coverage & chosen:
                raise ValueError("supplement repeats a main-run workflow")
            coverage |= chosen
        if coverage != {row["workflow_id"] for row in training_bundles}:
            raise ValueError("trained workflow IDs do not cover the training split")
        if {row["source_family_id"] for row in training_bundles} & {
            row["source_family_id"] for row in validation_bundles
        }:
            raise ValueError("training and validation source families overlap")
        if OUTPUT.exists():
            raise ValueError("validation output already exists; refuse to overwrite")
        OUTPUT.mkdir()
        arguments = [
            str(ROOT / ".venv/bin/python"),
            "-u",
            "code/scripts/evaluate_official_adapter.py",
            "--adapter",
            checkpoint,
            "--config",
            "configs/official_sft5090_compact5gpu_chunked.yaml",
            "--device",
            "cuda:3",
            "--token-budget",
            "131072",
            "--output-dir",
            str(OUTPUT.relative_to(ROOT)),
        ]
        with (OUTPUT / "evaluation.log").open("w") as log:
            child = subprocess.Popen(
                arguments,
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
                status="evaluating",
                child_pid=child.pid,
                command=arguments,
                checkpoint=checkpoint,
            )
            write_state(state)
            while child.poll() is None:
                write_state(state)
                time.sleep(10)
            if child.returncode:
                raise RuntimeError(
                    f"full validation exited {child.returncode}; inspect evaluation.log"
                )
        report = json.loads((OUTPUT / "report.json").read_text())
        if report["validation_evaluated"] != 23 or report["test_targets_loaded"]:
            raise ValueError("validation coverage or answer isolation failed")
        state.update(
            status="completed",
            validation_workflows=23,
            report=str((OUTPUT / "report.json").relative_to(ROOT)),
        )
        write_state(state)
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_state(state)
        raise


if __name__ == "__main__":
    main()
