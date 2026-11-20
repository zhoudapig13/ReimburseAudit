"""Run RNG-correct maximum-length check, full official SFT and full validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "runs/official_corrected_pipeline"
INITIAL = "runs/official_sft_extended/checkpoint_epoch_3"


def write_state(state):
    state["updated_unix"] = time.time()
    temporary = CONTROL / "state.tmp"
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(CONTROL / "state.json")


def run_stage(state, stage, arguments, directory):
    directory = ROOT / directory
    if directory.exists():
        raise ValueError(f"stage output already exists: {directory}")
    directory.mkdir()
    with (directory / "execution.log").open("w") as log:
        child = subprocess.Popen(
            [str(ROOT / ".venv/bin/python"), "-u", *arguments],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"),
        )
        state.update(
            status="running", stage=stage, child_pid=child.pid, command=arguments
        )
        write_state(state)
        while child.poll() is None:
            write_state(state)
            time.sleep(10)
        if child.returncode:
            raise RuntimeError(
                f"{stage} exited {child.returncode}; inspect {directory}/execution.log"
            )
    report = json.loads((directory / "report.json").read_text())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-verified-stability", action="store_true")
    args = parser.parse_args()
    CONTROL.mkdir(parents=True, exist_ok=True)
    if (CONTROL / "state.json").exists():
        previous = json.loads((CONTROL / "state.json").read_text())
        if (
            not args.reuse_verified_stability
            or previous["status"] != "failed"
            or previous.get("stage") != "corrected_maximum_length_stability"
        ):
            raise ValueError(
                "existing corrected controller is not eligible for verified-stage continuation"
            )
        for pid in [previous.get("controller_pid"), previous.get("child_pid")]:
            if (
                pid
                and Path(f"/proc/{pid}/cmdline").exists()
                and Path(f"/proc/{pid}/cmdline").read_bytes()
            ):
                raise ValueError(
                    "previous controller or training process is still live"
                )
        (CONTROL / "state_before_memory_budget_adjustment.json").write_text(
            json.dumps(previous, ensure_ascii=False, indent=2) + "\n"
        )
    state = {
        "controller_pid": os.getpid(),
        "status": "starting",
        "started_unix": time.time(),
        "previous_multigpu_weights_adopted": False,
        "initial_adapter": INITIAL,
    }
    write_state(state)
    try:
        old = json.loads(
            (ROOT / "runs/official_validation_queue/state.json").read_text()
        )
        if old["status"] not in {"completed", "failed"}:
            raise ValueError("previous GPU validation is still active")
        if args.reuse_verified_stability:
            folder = ROOT / "runs/official_corrected_stability"
            stability = json.loads((folder / "report.json").read_text())
            admission = json.loads(
                (folder / "memory_budget_admission.json").read_text()
            )
            config = yaml.safe_load(
                (ROOT / "configs/official_sft5090_corrected_stability.yaml").read_text()
            )
            previous_config = json.loads(json.dumps(stability["config"]))
            previous_config["environment"]["max_gpu_memory_gib"] = config[
                "environment"
            ]["max_gpu_memory_gib"]
            if (
                previous_config != config
                or admission["status"] != "passed_under_measured_hardware_budget"
            ):
                raise ValueError(
                    "verified stability configuration changed beyond admitted memory budget"
                )
            if (
                admission["adjusted_budget_gib"]
                != config["environment"]["max_gpu_memory_gib"]
            ):
                raise ValueError("memory admission does not match the configuration")
            for path, digest in stability["source_sha256"].items():
                if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
                    raise ValueError("source changed after verified stability test")
            state["verified_stability_reused"] = True
            write_state(state)
        else:
            stability = run_stage(
                state,
                "corrected_maximum_length_stability",
                [
                    "code/scripts/train_official_answers.py",
                    "--config",
                    "configs/official_sft5090_corrected_stability.yaml",
                    "--output-dir",
                    "runs/official_corrected_stability",
                    "--initial-adapter",
                    INITIAL,
                    "--device",
                    "cuda:3",
                    "--workflow-id",
                    "1.1.6",
                    "--epochs",
                    "20",
                ],
                "runs/official_corrected_stability",
            )
        if (
            stability["status"] != "passed"
            or stability["optimizer_steps"] != 20
            or stability["device_local_checkpoint_modules"] != 36
        ):
            raise ValueError("corrected maximum-length gate failed")
        budget = yaml.safe_load(
            (ROOT / "configs/official_sft5090_corrected.yaml").read_text()
        )["environment"]["max_gpu_memory_gib"]
        if max(stability["peak_gpu_by_device_gib"].values()) > budget:
            raise ValueError(
                "corrected maximum-length workload exceeds configured memory budget"
            )
        training = run_stage(
            state,
            "corrected_full_105_training",
            [
                "code/scripts/train_official_answers.py",
                "--config",
                "configs/official_sft5090_corrected.yaml",
                "--output-dir",
                "runs/official_corrected_train",
                "--initial-adapter",
                INITIAL,
                "--device",
                "cuda:3",
            ],
            "runs/official_corrected_train",
        )
        if (
            training["status"] != "passed"
            or training["selected_workflows"] != 105
            or training["deferred_workflows"]
            or training["device_local_checkpoint_modules"] != 36
        ):
            raise ValueError(
                "corrected training did not cover all 105 complete workflows"
            )
        if (
            training["source_truncated"]
            or training["validation_or_test_targets_loaded"]
        ):
            raise ValueError("corrected training source/answer isolation failed")
        validation = run_stage(
            state,
            "corrected_full_23_validation",
            [
                "code/scripts/evaluate_official_adapter.py",
                "--config",
                "configs/official_sft5090_corrected.yaml",
                "--adapter",
                "runs/official_corrected_train/checkpoint_epoch_3",
                "--device",
                "cuda:3",
                "--token-budget",
                "131072",
                "--output-dir",
                "runs/official_corrected_validation",
            ],
            "runs/official_corrected_validation",
        )
        if (
            validation["validation_evaluated"] != 23
            or validation["test_targets_loaded"]
        ):
            raise ValueError("corrected full validation coverage/isolation failed")
        state.update(
            status="completed",
            stage="corrected_pipeline_completed",
            training_workflows=105,
            validation_workflows=23,
            final_checkpoint="runs/official_corrected_train/checkpoint_epoch_3",
            validation_report="runs/official_corrected_validation/report.json",
        )
        write_state(state)
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_state(state)
        raise


if __name__ == "__main__":
    main()
