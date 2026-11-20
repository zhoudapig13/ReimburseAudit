"""Continue recorded reader shards and training when GPUs become available."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def alive(pid):
    try:
        return not Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].startswith("Z")
    except FileNotFoundError:
        return False


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    records = json.loads((run / "reader_processes.json").read_text())
    continuation = json.loads((run / "continuation.json").read_text())
    state = {"pid": os.getpid(), "stage": "continuing_readers", "automatic_accuracy_stop": False,
             "final_delivery_closed": False, "attempts": []}
    status_path = run / "queue_status.json"
    children = {}
    completed = set()
    adopted = {i: record["pid"] for i, record in enumerate(records) if alive(record["pid"])}

    def save(**values):
        state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(status_path, state)

    def output_records(record):
        command = record["command"]
        output = Path(command[command.index("--output") + 1])
        shard = int(command[command.index("--shard") + 1].split("/")[0])
        return output / f"shard_{shard:04d}"

    def ready(record):
        folder = output_records(record)
        selection = folder / "selection.json"
        if not selection.is_file():
            return False
        command = record["command"]
        index, total = map(int, command[command.index("--shard") + 1].split("/"))
        expected = set(json.loads(selection.read_text())["workflows"][index::total])
        observed = {}
        for path in folder.glob("*.json"):
            if path.name == "selection.json":
                continue
            row = json.loads(path.read_text())
            if "workflow_id" in row:
                observed[row["workflow_id"]] = row
        failed = [row for row in observed.values() if row.get("status") == "failed"]
        if failed:
            save(stage="reader_data_requires_repair", reader_failures=failed)
            raise ValueError("reader source errors need repair; records retained")
        return expected.issubset(observed)

    while len(completed) < len(records):
        for i, pid in list(adopted.items()):
            if not alive(pid):
                del adopted[i]
        for i, child in list(children.items()):
            if child.poll() is not None:
                del children[i]
                for attempt in reversed(state["attempts"]):
                    if attempt["pid"] == child.pid:
                        attempt["exit_code"] = child.returncode
                        break
        running = set(adopted) | set(children)
        for i, record in enumerate(records):
            if i not in running and i not in completed and ready(record):
                completed.add(i)
        if len(completed) == len(records):
            break
        result = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"
        ], text=True)
        occupied = {records[i]["gpu"] for i in running}
        available = [int(parts[0]) for line in result.splitlines()
                     if len(parts := [part.strip() for part in line.split(",")]) == 3
                     and int(parts[1]) >= 24576 and int(parts[2]) <= 10
                     and int(parts[0]) not in occupied]
        for i, record in enumerate(records):
            if i in running or i in completed or not available:
                continue
            attempts = sum(row["shard_index"] == i for row in state["attempts"])
            if attempts >= 3:
                save(stage="reader_launch_requires_repair", failed_shard=i)
                return 2
            gpu = available.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            with Path(record["log"]).open("ab") as stream:
                child = subprocess.Popen(record["command"], env=env, stdout=stream,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            children[i] = child
            record.update(pid=child.pid, gpu=gpu)
            state["attempts"].append({"pid": child.pid, "gpu": gpu, "shard_index": i,
                                      "started_at": datetime.now(timezone.utc).isoformat()})
            atomic_json(run / "reader_processes.json", records)
        save(stage="reading_materials" if adopted or children else "waiting_for_available_gpu",
             completed_shards=sorted(completed), total_shards=len(records),
             running_pids=[*adopted.values(), *(child.pid for child in children.values())])
        time.sleep(10)
    save(stage="readers_complete_starting_three_module_training", completed_shards=sorted(completed))
    with (run / "pipeline.log").open("ab") as stream:
        child = subprocess.Popen(continuation["pipeline_command"], stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    save(stage="three_module_pipeline_running", pipeline_pid=child.pid)
    code = child.wait()
    training = run / "reviewed_pilot_v3/pipeline_status.json"
    training_state = json.loads(training.read_text()) if training.is_file() else {}
    save(stage="three_module_checkpoint_ready" if code == 0 else "training_requires_repair",
         pipeline_exit_code=code, training=training_state)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
