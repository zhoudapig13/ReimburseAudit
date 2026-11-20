"""Train eight hours, validate all workflows, then continue until a plateau."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
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

from audit_engine.training.continuous import plateau_decision  # noqa: E402


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-hours", type=float, default=8)
    parser.add_argument("--later-hours", type=float, default=2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--learning-rate-scale", type=float)
    parser.add_argument(
        "--training-objective",
        choices=["standard", "decision_amount_balanced"],
        default="standard",
    )
    args = parser.parse_args()
    if args.first_hours <= 0 or args.later_hours <= 0 or args.patience < 1:
        parser.error("durations and patience must be positive")
    if args.resume_checkpoint and not args.resume_run:
        parser.error("resume-checkpoint requires resume-run")
    if args.learning_rate_scale is not None and args.learning_rate_scale <= 0:
        parser.error("learning-rate scale must be positive")
    folder = args.output_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "runs/continuous_official_training.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder / "state.json").exists():
        raise ValueError("existing run state; reconcile before restarting")
    config = "configs/official_sft5090_corrected.yaml"
    initial = ROOT / "runs/official_sft_extended/checkpoint_epoch_3"
    previous = None
    if args.resume_run:
        previous = json.loads((args.resume_run / "state.json").read_text())
        if (
            previous["status"] not in {"completed", "stopped_by_user"}
            or not previous["rounds"]
        ):
            raise ValueError("resume run must have completed training and validation")
        initial = args.resume_checkpoint or Path(previous["rounds"][-1]["checkpoint"])
        if not (initial / "training_state.pt").is_file():
            raise ValueError("full training checkpoint missing")
    stability = json.loads(
        (ROOT / "runs/official_corrected_stability/report.json").read_text()
    )
    if (
        stability["status"] != "passed"
        or stability["optimizer_steps"] != 20
        or stability["device_local_checkpoint_modules"] != 36
    ):
        raise ValueError("corrected maximum-length prerequisite missing")
    for path, digest in stability["source_sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise ValueError("sources changed since full-input stability check")
    state = {
        "controller_pid": os.getpid(),
        "status": "starting",
        "started_unix": time.time(),
        "first_training_hours": args.first_hours,
        "schedule_hours": previous.get(
            "schedule_hours", previous["first_training_hours"]
        )
        if previous
        else args.first_hours,
        "subsequent_training_hours": args.later_hours,
        "patience_per_learning_rate": args.patience,
        "config": config,
        "training_objective": args.training_objective,
        "initial_adapter": str(initial),
        "rounds": [],
        "stop_rule": "three evaluations without improvement, halve learning rate, then three further evaluations without improvement",
        "comparison": "joint correct FLAG and amount with valid schema; ties use field accuracy, schema coverage, explanation exact match",
    }
    if previous:
        state.update(
            resumed_from_run=str(args.resume_run.resolve()),
            best_checkpoint=previous["best_checkpoint"],
            best_metrics=previous["best_metrics"],
            base_metrics=previous["base_metrics"],
        )

    def save():
        state["updated_unix"] = time.time()
        write_json(folder / "state.json", state)

    def run_stage(stage, arguments, output):
        output.mkdir()
        with (output / "execution.log").open("w") as log:
            child = subprocess.Popen(
                [str(ROOT / ".venv/bin/python"), "-u", *arguments],
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
                status="running", stage=stage, child_pid=child.pid, command=arguments
            )
            save()
            while child.poll() is None:
                save()
                time.sleep(10)
            state["child_returncode"] = child.returncode
            save()
            if child.returncode:
                raise RuntimeError(f"{stage} exited {child.returncode}; see {output}")
        return json.loads((output / "report.json").read_text())

    save()
    try:
        best = previous["best_metrics"] if previous else None
        stale = 0
        reduced = False
        scale = previous.get("learning_rate_scale", 1.0) if previous else 1.0
        if args.learning_rate_scale is not None:
            scale = args.learning_rate_scale
        schedule_hours = state["schedule_hours"]
        state.update(learning_rate_scale=scale, stale_rounds=stale, reduced=reduced)
        checkpoint = initial
        cycle = 1
        while True:
            state["cycle"] = cycle
            duration = args.first_hours if cycle == 1 else args.later_hours
            train_dir = folder / f"cycle_{cycle:03d}_train"
            command = [
                "code/scripts/train_official_answers.py",
                "--config",
                config,
                "--output-dir",
                str(train_dir),
                "--device",
                "cuda:3",
                "--training-seconds",
                str(duration * 3600),
                "--schedule-seconds",
                str(schedule_hours * 3600),
                "--learning-rate-scale",
                str(scale),
                "--skip-train-loss-evaluation",
                "--require-all-workflows",
                "--training-objective",
                args.training_objective,
                "--resume-checkpoint" if previous or cycle > 1 else "--initial-adapter",
                str(checkpoint),
            ]
            trained = run_stage(f"cycle_{cycle:03d}_training", command, train_dir)
            if (
                trained["status"] != "passed"
                or trained["selected_workflows"] != trained["train_workflows_available"]
                or trained["deferred_workflows"]
                or trained["source_truncated"]
                or trained["validation_or_test_targets_loaded"]
                or trained["device_local_checkpoint_modules"] != 36
                or trained["training_seconds_this_run"] < duration * 3600
                or trained["optimizer_steps_this_run"] < 1
            ):
                raise ValueError(
                    "full-input duration/optimization/isolation gate failed"
                )
            checkpoint = Path(trained["final_checkpoint"])
            validation_dir = folder / f"cycle_{cycle:03d}_validation"
            command = [
                "code/scripts/evaluate_official_adapter.py",
                "--config",
                config,
                "--adapter",
                str(checkpoint),
                "--device",
                "cuda:3",
                "--token-budget",
                "131072",
                "--output-dir",
                str(validation_dir),
            ]
            if previous or cycle > 1:
                command.append("--adapter-only")
            report = run_stage(f"cycle_{cycle:03d}_validation", command, validation_dir)
            if (
                report["validation_evaluated"] != report["validation_available"]
                or report["validation_evaluated"] != 23
                or report["deferred_workflow_ids"]
                or report["test_targets_loaded"]
            ):
                raise ValueError("full validation coverage/isolation gate failed")
            if cycle == 1 and not previous:
                state["base_metrics"] = report["base"]
            current = report["adapter"]
            decision = plateau_decision(best, current, stale, reduced, args.patience)
            stale = decision["stale_rounds"]
            if decision["action"] == "improved":
                best = current
                state.update(best_checkpoint=str(checkpoint), best_metrics=current)
            summary = {
                "cycle": cycle,
                "completed_beijing": datetime.now(
                    ZoneInfo("Asia/Shanghai")
                ).isoformat(),
                "training_hours_this_round": trained["training_seconds_this_run"]
                / 3600,
                "training_hours_total": trained["training_seconds_total"] / 3600,
                "optimizer_steps": trained["optimizer_steps"],
                "validation_workflows": report["validation_evaluated"],
                "adapter": current,
                "base": state["base_metrics"],
                "best": best,
                "checkpoint": str(checkpoint),
                "report": str(validation_dir / "report.json"),
                "action": decision["action"],
                "explanation_metric": "exact text match, not semantic grading",
            }
            state["rounds"].append(summary)
            write_json(folder / f"result_{cycle:03d}.json", summary)
            write_json(folder / "latest_result.json", summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            if decision["action"] == "stop":
                state.update(
                    status="completed",
                    stage="validation_plateau",
                    finished_unix=time.time(),
                )
                save()
                break
            if decision["action"] == "reduce_learning_rate":
                scale *= 0.5
                reduced = True
            state.update(stale_rounds=stale, learning_rate_scale=scale, reduced=reduced)
            save()
            cycle += 1
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
