"""Finish the new frozen-model evaluation and export the fixed comparison."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "runs/three_model_comparison_20261020"


def main():
    state = {
        "pid": os.getpid(),
        "stage": "waiting_for_model_download",
        "status": "running",
    }

    def save():
        state["updated_unix"] = time.time()
        path = FOLDER / "comparison_state.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)

    def run(stage, arguments):
        state["stage"] = stage
        with (FOLDER / f"{stage}.log").open("w") as log:
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
            state["child_pid"] = child.pid
            save()
            while child.poll() is None:
                save()
                time.sleep(5)
            if child.returncode:
                raise RuntimeError(
                    f"{stage} exited {child.returncode}; see {stage}.log"
                )

    save()
    try:
        tokenizer = ROOT / "models/GLM-4.1V-9B-Thinking/tokenizer.json"
        while not tokenizer.exists():
            bootstrap = FOLDER / "glm_tokenizer_bootstrap/tokenizer.json"
            if bootstrap.exists():
                manifest = json.loads(
                    (tokenizer.parent / ".range_parts/manifest.json").read_text()
                )
                expected = next(
                    item
                    for item in manifest["siblings"]
                    if item["rfilename"] == "tokenizer.json"
                )
                if (
                    hashlib.sha256(bootstrap.read_bytes()).hexdigest()
                    != expected["lfs"]["sha256"]
                ):
                    raise ValueError("tokenizer bootstrap checksum mismatch")
                shutil.copyfile(bootstrap, tokenizer)
                break
            state["stage"] = "waiting_for_tokenizer"
            save()
            time.sleep(5)
        run(
            "glm_preflight",
            [
                "code/scripts/evaluate_frozen_glm41v.py",
                "--prepare-only",
                "--output-dir",
                str(FOLDER / "glm_preflight"),
            ],
        )
        state["stage"] = "waiting_for_model_download"
        download_restarts = 0
        while not (FOLDER / "new_model_download.json").exists():
            launch = json.loads((FOLDER / "new_model_download_launch.json").read_text())
            stat = Path(f"/proc/{launch['pid']}/stat")
            if not stat.exists() or stat.read_text().split()[2] == "Z":
                if download_restarts >= 8:
                    raise RuntimeError(
                        "model download failed after eight resumable restarts"
                    )
                command = launch["command"]
                if "configs/glm41v_comparison.yaml" not in command:
                    raise ValueError("download restart model mismatch")
                with (FOLDER / "glm_download.log").open("a") as log:
                    child = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                path = FOLDER / "new_model_download_launch.json"
                temporary = path.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps({"pid": child.pid, "command": command}, indent=2) + "\n"
                )
                temporary.replace(path)
                download_restarts += 1
                state["download_restarts"] = download_restarts
            save()
            time.sleep(5)
        while True:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            occupied = [
                int(line.split(",")[0])
                for line in result.stdout.splitlines()
                if int(line.split(",")[0]) in (4, 5, 6)
                and int(line.split(",")[1]) > 512
            ]
            if not occupied:
                break
            state.update(stage="waiting_for_free_gpus", occupied_gpus=occupied)
            save()
            time.sleep(30)
        run(
            "new_model_evaluation",
            [
                "code/scripts/evaluate_frozen_glm41v.py",
                "--output-dir",
                str(FOLDER / "new_model"),
            ],
        )
        if not (FOLDER / "qwen/report.json").exists():
            raise RuntimeError("Qwen comparison results missing")
        run("export", ["code/scripts/export_three_model_comparison.py"])
        state.update(
            status="completed", stage="comparison_complete", finished_unix=time.time()
        )
        save()
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
