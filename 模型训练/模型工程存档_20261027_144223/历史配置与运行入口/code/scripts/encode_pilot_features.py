"""Encode a non-overlapping shard of train-only pilot material views."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from audit_engine.readers.qwen_encoder import FrozenQwenEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/smoke5090.yaml"))
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/Qwen3-VL-8B-Instruct")
    )
    parser.add_argument(
        "--jobs", type=Path, default=Path("data/features_pilot/encoding_jobs.jsonl")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/features_pilot/encoded")
    )
    parser.add_argument("--shard", default="0/2")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--wait-pid", type=int)
    args = parser.parse_args()
    index, total = map(int, args.shard.split("/"))
    if not 0 <= index < total:
        raise ValueError("invalid shard selection")
    if args.wait_pid:
        process = Path(f"/proc/{args.wait_pid}")
        if (
            process.exists()
            and b"initial_training.py" not in (process / "cmdline").read_bytes()
        ):
            raise ValueError("wait PID is not the initial smoke process")
        print("waiting_for_real_backbone_smoke", flush=True)
        while process.exists():
            try:
                if (process / "stat").read_text().split(") ", 1)[1].startswith("Z"):
                    break
            except FileNotFoundError:
                break
            time.sleep(10)
    report_path = Path("runs/smoke5090/backbone_report.json")
    if not report_path.is_file():
        print("upstream_backbone_smoke_not_completed", flush=True)
        raise SystemExit(2)
    report = json.loads(report_path.read_text())
    pipeline = json.loads(Path("runs/smoke5090/pipeline_state.json").read_text())
    if (
        report["status"] != "passed"
        or not report["within_gpu_memory_budget"]
        or pipeline["stage"] != "initial_smoke_and_optimization_passed"
    ):
        raise ValueError("actual initial smoke and optimization have not passed")
    config = yaml.safe_load(args.config.read_text())
    if report["revision"] != config["model"]["revision"]:
        raise ValueError("smoke model revision mismatch")
    planned = [json.loads(line) for line in args.jobs.open()]
    selected = planned[index::total]
    if any(
        job["split"] != "train" or job["base_revision"] != config["model"]["revision"]
        for job in selected
    ):
        raise ValueError("job split or base revision mismatch")
    output = args.output_root / f"shard_{index:04d}"
    output.mkdir(parents=True, exist_ok=True)
    encoder = FrozenQwenEncoder(config, args.model_path, args.device)
    torch.set_num_threads(8)
    root = Path(config["project"]["data_root"]).resolve()
    verified_sources = {}
    started = time.monotonic()
    records = []
    for number, job in enumerate(selected):
        source = (root / job["raw_ref"]).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError("unsafe or missing evidence source")
        if source not in verified_sources:
            verified_sources[source] = hashlib.sha256(source.read_bytes()).hexdigest()
        if verified_sources[source] != job["source_sha256"]:
            raise ValueError("raw material changed after feature planning")
        image = source if job["kind"] == "original_image" else None
        key = encoder.cache_key(job["text"], image)
        path = output / f"{key}.pt"
        if path.exists():
            stored = torch.load(path, map_location="cpu", weights_only=True)
            vector = stored["vector"]
            if stored["signature"] != encoder.signature or stored["cache_key"] != key:
                raise ValueError("cached feature identity mismatch")
        else:
            vector = encoder.encode(job["text"], image)
            temporary = path.with_suffix(".tmp")
            torch.save(
                {"vector": vector, "signature": encoder.signature, "cache_key": key},
                temporary,
            )
            temporary.replace(path)
        if vector.shape != (4096,) or not torch.isfinite(vector).all():
            raise ValueError("invalid frozen feature tensor")
        records.append(
            {
                "workflow_id": job["workflow_id"],
                "evidence_id": job["evidence_id"],
                "view_id": job["view_id"],
                "view_count": job["view_count"],
                "source_sha256": job["source_sha256"],
                "path": path.as_posix(),
                "cache_key": key,
            }
        )
        if (number + 1) % 10 == 0 or number + 1 == len(selected):
            print(
                json.dumps(
                    {
                        "shard": args.shard,
                        "completed": number + 1,
                        "total": len(selected),
                        "seconds": round(time.monotonic() - started),
                        "peak_gpu_gib": torch.cuda.max_memory_allocated(args.device)
                        / 1024**3,
                    }
                ),
                flush=True,
            )
    for source, digest in verified_sources.items():
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError("source changed during feature extraction")
    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    summary = {
        "status": "passed",
        "shard": args.shard,
        "jobs": len(records),
        "feature_dimension": 4096,
        "projection_applied": False,
        "labels_read": False,
        "base_signature": encoder.signature,
        "peak_gpu_gib": torch.cuda.max_memory_allocated(args.device) / 1024**3,
        "elapsed_seconds": time.monotonic() - started,
    }
    if summary["peak_gpu_gib"] > config["environment"]["max_gpu_memory_gib"]:
        summary["status"] = "memory_budget_exceeded"
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
