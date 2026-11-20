"""Check every planned pilot input with the actual processor, without weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoProcessor

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
        "--output",
        type=Path,
        default=Path("runs/bootstrap/pilot_processor_preflight.json"),
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = yaml.safe_load(args.config.read_text())
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=True,
        max_pixels=config["reader"]["image_max_pixels"],
    )
    processor.image_processor.size["longest_edge"] = config["reader"][
        "image_max_pixels"
    ]
    # Only call the shared input adapter. No checkpoint/model object is loaded,
    # and this diagnostic cannot authorize backbone use or training.
    adapter = FrozenQwenEncoder.__new__(FrozenQwenEncoder)
    adapter.config, adapter.processor, adapter.device = config, processor, "cpu"
    root = Path(config["project"]["data_root"]).resolve()
    records = []
    for line in args.jobs.open():
        job = json.loads(line)
        if (
            job["split"] != "train"
            or job["base_revision"] != config["model"]["revision"]
        ):
            raise ValueError("unexpected job split or base revision")
        source = (root / job["raw_ref"]).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError("unsafe or missing material source")
        image = source if job["kind"] == "original_image" else None
        text = (
            "Read this material and preserve factual information. Do not make a final audit decision.\n"
            + job["text"]
        )
        inputs = adapter.inputs(text, image)
        pixels = 0
        if "image_grid_thw" in inputs:
            pixels = (
                int(inputs.image_grid_thw[:, 1:].prod(-1).max())
                * processor.image_processor.patch_size**2
            )
        records.append(
            {
                "workflow_id": job["workflow_id"],
                "evidence_id": job["evidence_id"],
                "view_id": job["view_id"],
                "kind": job["kind"],
                "input_tokens": inputs.input_ids.shape[-1],
                "resized_pixels": pixels,
            }
        )
        if len(records) % 50 == 0:
            print(json.dumps({"checked": len(records)}), flush=True)
    if not records:
        raise ValueError("empty job plan")
    report = {
        "status": "passed",
        "scope": "processor_input_limits_only",
        "model_loaded": False,
        "formal_training": False,
        "labels_read": False,
        "revision": config["model"]["revision"],
        "jobs": len(records),
        "original_image_jobs": sum(r["kind"] == "original_image" for r in records),
        "token_budget": config["reader"]["total_input_tokens"],
        "pixel_budget": config["reader"]["image_max_pixels"],
        "largest_token_input": max(records, key=lambda r: r["input_tokens"]),
        "largest_pixel_input": max(records, key=lambda r: r["resized_pixels"]),
        "inputs": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "inputs"}), flush=True)


if __name__ == "__main__":
    main()
