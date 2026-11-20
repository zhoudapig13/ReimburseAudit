"""Download/verify, exercise the real backbone, then optimize the audit heads."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-pid", type=int)
    parser.add_argument("--verified-checkpoint", action="store_true")
    parser.add_argument("--config", default="configs/smoke5090.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/smoke5090"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_file = args.output_dir / "pipeline_state.json"

    def state(stage):
        state_file.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "stage": stage,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "download_pid": args.download_pid,
                },
                indent=2,
            )
        )
        print(stage, flush=True)

    if args.verified_checkpoint:
        state("checking_verified_checkpoint")
    elif args.download_pid:
        state("waiting_for_checkpoint_process")
        process = Path(f"/proc/{args.download_pid}")
        if (
            process.exists()
            and b"download_checkpoint.py" not in (process / "cmdline").read_bytes()
        ):
            raise RuntimeError("download PID does not identify the expected process")
        while process.exists():
            try:
                stat = (process / "stat").read_text()
            except FileNotFoundError:
                break
            if stat.split(") ", 1)[1].startswith("Z"):
                break
            time.sleep(10)
    else:
        state("downloading_checkpoint")
        subprocess.run(
            [sys.executable, "code/scripts/download_checkpoint.py"], check=True
        )
    download_path = Path("runs/bootstrap/model_download.json")
    if not download_path.is_file():
        state("checkpoint_download_failed")
        raise RuntimeError(
            "downloader terminated before checkpoint verification; inspect checkpoint_download.log"
        )
    download = json.loads(download_path.read_text())
    if not download.get("all_shard_sha256_verified"):
        raise RuntimeError("checkpoint hash verification has not completed")
    state("running_real_backbone_smoke")
    subprocess.run(
        [
            sys.executable,
            "code/scripts/smoke_qwen.py",
            "--config",
            args.config,
            "--output-dir",
            str(args.output_dir),
        ],
        check=True,
    )
    state("training_heads_on_real_backbone_features")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "audit_engine.training.smoke",
            "--config",
            args.config,
            "--features",
            str(args.output_dir / "fixture_features.pt"),
            "--output-dir",
            str(args.output_dir / "heads"),
            "--steps",
            "80",
        ],
        check=True,
    )
    backbone = json.loads((args.output_dir / "backbone_report.json").read_text())
    heads = json.loads((args.output_dir / "heads/report.json").read_text())
    if backbone["status"] != "passed" or not heads["all_stage_losses_decreased"]:
        raise RuntimeError("initial training acceptance failed")
    lock = {
        "scope": "verified_smoke_environment_only",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            d.metadata["Name"]: d.version
            for d in metadata.distributions()
            if d.metadata["Name"]
        },
        "model": download,
        "backbone_smoke": backbone,
        "heads_smoke": heads,
        "formal_training_approved": False,
    }
    (args.output_dir / "environment.smoke.lock.json").write_text(
        json.dumps(lock, indent=2)
    )
    state("initial_smoke_and_optimization_passed")


if __name__ == "__main__":
    main()
