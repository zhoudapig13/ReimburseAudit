"""Refresh real variant encodings, run B4, then continue until user cutoff."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    root = Path.cwd()
    run = root / "runs/patent_full_20261027/reviewed_full_v4"
    run.mkdir(parents=True, exist_ok=True)
    cutoff = datetime.fromisoformat("2026-10-27T07:00:00+08:00").timestamp()
    records = json.loads(Path("runs/patent_adaptation_20261027/policy_proposal_processes.json").read_text())
    proposal = records[0]
    status = {"pid": os.getpid(), "stage": "preparing_source_verified_variants", "deadline": "2026-10-27T07:00:00+08:00"}

    def save(**values):
        status.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        temp = run / "pipeline_status.tmp"
        temp.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
        temp.replace(run / "pipeline_status.json")

    def start(command, name, gpu=None):
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with (run / (name + ".log")).open("ab") as stream:
            child = subprocess.Popen([sys.executable, "-u", *map(str, command)], env=env,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        save(job={"pid": child.pid, "gpu": gpu, "name": name})
        return child

    save()
    for name in ("reader_cases.jsonl", "annotations.jsonl", "augmentation_manifest.json"):
        shutil.copyfile(Path("data/patent_verified_deletions_v1") / name, run / name)
    shutil.copytree("runs/patent_full_20261027/reviewed_pilot_v3/features", run / "features", dirs_exist_ok=True)
    # Temporarily release only this run's own policy-reader shard; it resumes
    # from its saved calls after real feature encoding releases the same GPU.
    try:
        arguments = Path(f"/proc/{proposal['pid']}/cmdline").read_bytes().decode(errors="replace")
        if "read_patent_cases.py" in arguments and "patent_validation_policy_proposals_v1" in arguments:
            os.kill(proposal["pid"], signal.SIGTERM)
    except FileNotFoundError:
        pass
    gpu = proposal["gpu"]
    while time.time() < cutoff:
        rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        ready = any(int(parts[0]) == gpu and int(parts[1]) >= 24576 and int(parts[2]) <= 10
                    for line in rows.splitlines() if len(parts := line.split(",")) == 3)
        if ready:
            break
        save(stage="waiting_for_reserved_encoding_gpu")
        time.sleep(2)
    else:
        save(stage="user_deadline_reached_before_encoding")
        return 2
    save(stage="encoding_originals_and_actual_file_deletions")
    child = start(["code/scripts/patent_pipeline.py", "encode", "--cases", run / "reader_cases.jsonl",
                   "--features", run / "features", "--config", "configs/patent_full_components.yaml",
                   "--model-path", "/mnt/public/zhouyibo/.cache/reimbursement-patent/model/Qwen3-VL-8B-Instruct"], "encode", gpu)
    code = child.wait()
    with Path(proposal["log"]).open("ab") as stream:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        resumed = subprocess.Popen(proposal["command"], env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    proposal["pid"] = resumed.pid
    Path("runs/patent_adaptation_20261027/policy_proposal_processes.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
    if code:
        save(stage="encoding_requires_repair", exit_code=code)
        return 3
    if time.time() >= cutoff:
        save(stage="user_deadline_reached_before_training")
        return 2
    save(stage="training_B1_B2_B3_B4_C")
    child = start(["code/scripts/patent_pipeline.py", "train", "--cases", run / "reader_cases.jsonl",
                   "--annotations", run / "annotations.jsonl", "--features", run / "features",
                   "--config", "configs/patent_full_components.yaml", "--output", run / "weights",
                   "--initial-checkpoint", "runs/patent_full_20261027/reviewed_pilot_v3/weights/audit_heads.pt",
                   "--device", "cpu"], "train")
    if child.wait():
        save(stage="training_requires_repair")
        return 4
    summary = json.loads((run / "weights/training_summary.json").read_text())
    if not summary["three_components_optimized"] or summary["stage_steps"].get("B4", 0) == 0:
        save(stage="required_component_training_incomplete")
        return 5
    save(stage="all_required_stages_checkpoint_ready", checkpoint=str(run / "weights/audit_heads.pt"),
         stage_steps=summary["stage_steps"], nonzero_gradient_steps=summary["nonzero_gradient_steps"],
         original_training_workflows=summary["original_training_workflows"],
         source_verified_training_variants=summary["source_verified_training_variants"], final_delivery_closed=False)
    old = json.loads(Path("runs/patent_full_20261027/continuous_heads/state.json").read_text())["pid"]
    try:
        args = Path(f"/proc/{old}/cmdline").read_bytes().decode(errors="replace")
        if "train_heads_until.py" in args and "continuous_heads" in args:
            os.kill(old, signal.SIGTERM)
    except FileNotFoundError:
        pass
    if time.time() < cutoff:
        child = start(["code/scripts/train_heads_until.py", "--input-run", run,
                       "--output", "runs/patent_full_20261027/continuous_full_components",
                       "--deadline", "2026-10-27T07:00:00+08:00", "--device", "cpu"], "continuous")
        save(continuous_training_pid=child.pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
