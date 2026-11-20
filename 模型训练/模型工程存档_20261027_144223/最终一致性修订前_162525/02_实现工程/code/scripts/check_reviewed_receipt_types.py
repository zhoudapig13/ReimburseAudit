"""Run source reading and all patent heads for reviewed invoice-type clauses."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    root = Path("runs/patent_adaptation_20261027/receipt_type_check")
    root.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    model = "/mnt/public/zhouyibo/.cache/reimbursement-patent/model/Qwen3-VL-8B-Instruct"

    def run(arguments, name):
        with (root / (name + ".log")).open("ab") as log:
            subprocess.run([python, "-u", *map(str, arguments)], stdout=log,
                           stderr=subprocess.STDOUT, check=True)

    try:
        run(["code/scripts/read_patent_cases.py", "--mode", "read", "--split", "validation",
             "--cases", "data/patent_reviewed/policy_cases_v8.jsonl",
             "--workflow-list", "runs/patent_adaptation_20261027/receipt_type_selection.json",
             "--config", "configs/patent_full_components.yaml", "--model-path", model,
             "--output", root / "reader"], "read")
        records = [json.loads(p.read_text()) for p in (root / "reader").glob("shard_*/*.json")
                   if p.name != "selection.json"]
        if len(records) != 2 or any("case" not in record for record in records):
            raise ValueError("incomplete actual receipt readings")
        cases = root / "cases.jsonl"
        cases.write_text("".join(json.dumps(record["case"], ensure_ascii=False) + "\n"
                                 for record in records))
        cache = root / "features/material_cache"
        shutil.copytree("runs/patent_eval_full_20261027_0616/features/material_cache", cache,
                        dirs_exist_ok=True)
        run(["code/scripts/patent_pipeline.py", "encode", "--cases", cases,
             "--features", root / "features", "--config", "configs/patent_full_components.yaml",
             "--model-path", model], "encode")
        run(["code/scripts/patent_pipeline.py", "predict", "--cases", cases,
             "--features", root / "features", "--checkpoint",
             "runs/patent_full_20261027/reviewed_full_v5/weights/audit_heads.pt",
             "--device", "cpu", "--output", root / "predictions.jsonl"], "predict")
        (root / "status.json").write_text(json.dumps({"stage": "integration_complete",
            "official_reference_answers_opened": False, "workflows": 2}))
    except Exception as error:
        (root / "status.json").write_text(json.dumps({"stage": "failed", "error": str(error)}))
        raise
    finally:
        # Resume the interrupted source-policy reader on the same GPU. Its
        # committed call records allow the unfinished source view to be retried.
        previous = json.loads(Path("runs/patent_adaptation_20261027/policy_atom_processes.json").read_text())[-1]
        with Path(previous["log"]).open("ab") as log:
            child = subprocess.Popen(previous["command"], stdout=log, stderr=subprocess.STDOUT,
                                     env=os.environ.copy(), start_new_session=True)
        previous["pid"] = child.pid
        Path("runs/patent_adaptation_20261027/policy_reader_gpu5_resumed.json").write_text(json.dumps(previous, indent=2))


if __name__ == "__main__":
    main()
