"""Resolve recorded image schema failures with explicit source/call provenance."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from audit_engine.readers.qwen_encoder import FrozenQwenEncoder
from audit_engine.readers.visual import validate_visual_output


def atomic_json(path: Path, value: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    torch.set_num_threads(4)
    source_path = Path("data/reader_pilot_v2/qwen_image_readings.jsonl")
    previous = [json.loads(line) for line in source_path.open()]
    failed = [r for r in previous if r["schema_status"] == "failed"]
    config = yaml.safe_load(Path("configs/patent_reader_delivery.yaml").read_text())
    config["reader"]["image_max_pixels"] = 1048576
    root = Path(config["project"]["data_root"]).resolve()
    output = Path("data/reader_pilot_repairs")
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()

    def state(status, **extra):
        atomic_json(
            output / "state.json",
            {
                "pid": os.getpid(),
                "status": status,
                "started_at": started,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                **extra,
            },
        )

    state("starting", failures=len(failed))
    encoder = None
    records = []
    prompt = (
        runpy.run_path("code/scripts/read_pilot_images.py")["PROMPT"]
        + "\nReturn compact JSON. Keep source_text to a short exact label/value supporting each fact. Preserve each genuinely different line item once. Do not repeat the same fact."
    )
    try:
        for parent in failed:
            source = (root / parent["raw_ref"]).resolve()
            if (
                not source.is_relative_to(root)
                or hashlib.sha256(source.read_bytes()).hexdigest()
                != parent["source_sha256"]
            ):
                raise ValueError("repair source unsafe or changed")
            candidate = None
            raw_outputs = []
            method = "schema_revalidation_only"
            signature = parent["signature"]
            error = None
            for raw in reversed(parent["raw_outputs"]):
                try:
                    candidate = validate_visual_output(raw)
                    break
                except (ValueError, TypeError):
                    pass
            if candidate is None:
                if encoder is None:
                    encoder = FrozenQwenEncoder(
                        config, Path("models/Qwen3-VL-8B-Instruct"), "cuda:4"
                    )
                method = "one_high_resolution_reread"
                signature = {
                    **encoder.signature,
                    "reading_prompt_sha256": hashlib.sha256(
                        prompt.encode()
                    ).hexdigest(),
                    "max_new_tokens": 4096,
                    "reading_version": "qwen_pilot_visual_repair_v3",
                }
                raw = encoder.read(prompt, source, 4096)
                raw_outputs.append(raw)
                try:
                    candidate = validate_visual_output(raw)
                except (ValueError, TypeError) as exc:
                    error = str(exc)
            if (
                hashlib.sha256(source.read_bytes()).hexdigest()
                != parent["source_sha256"]
            ):
                raise ValueError("repair source changed during reading")
            record = {
                k: parent[k]
                for k in ("workflow_id", "evidence_id", "raw_ref", "source_sha256")
            }
            record.update(
                candidate=candidate,
                signature=signature,
                repair_method=method,
                parent_cache_key=parent["cache_key"],
                parent_schema_error=parent["schema_error"],
                parent_source=source_path.as_posix(),
                raw_outputs=raw_outputs,
                schema_status="passed" if candidate else "failed",
                schema_error=error,
                review_status="pending_human_review",
                approved_training_label=False,
                visual_fact_accuracy="pending_review",
            )
            records.append(record)
            atomic_json(output / f"{parent['cache_key']}.json", record)
            state("repairing", completed=len(records), total=len(failed))
            print(
                json.dumps(
                    {
                        "workflow": record["workflow_id"],
                        "evidence": record["evidence_id"],
                        "method": method,
                        "schema_status": record["schema_status"],
                    }
                ),
                flush=True,
            )
        temporary = output / "repair_records.jsonl.tmp"
        temporary.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        )
        temporary.replace(output / "repair_records.jsonl")
        report = {
            "status": "completed_with_pending_review",
            "parent_input_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "failed_parents": len(failed),
            "resolved_schema_failures": sum(
                r["schema_status"] == "passed" for r in records
            ),
            "new_model_calls": sum(len(r["raw_outputs"]) for r in records),
            "high_resolution_pixel_budget": 1048576,
            "peak_gpu_gib": torch.cuda.max_memory_allocated("cuda:4") / 1024**3
            if encoder
            else None,
            "approved_labels": 0,
            "source_coverage": "recorded_failed_images_only",
            "started_at": started,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(output / "stage_manifest.json", report)
        state("completed_with_pending_review")
        print(json.dumps(report), flush=True)
    except Exception as exc:
        state("failed", error_type=type(exc).__name__, message=str(exc))
        raise


if __name__ == "__main__":
    main()
