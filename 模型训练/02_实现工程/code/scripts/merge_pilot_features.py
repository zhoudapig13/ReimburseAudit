"""Verify complete shard coverage and pool all views into file-level vectors."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs", type=Path, default=Path("data/features_pilot/encoding_jobs.jsonl")
    )
    parser.add_argument(
        "--encoded-root", type=Path, default=Path("data/features_pilot/encoded")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/features_pilot/pooled")
    )
    parser.add_argument("--shards", type=int, default=2)
    args = parser.parse_args()
    planned = {
        (job["workflow_id"], job["evidence_id"], job["view_id"]): job
        for job in (json.loads(line) for line in args.jobs.open())
    }
    if not planned:
        raise ValueError("empty feature plan")
    records = {}
    signature = None
    for index in range(args.shards):
        shard = args.encoded_root / f"shard_{index:04d}"
        summary = json.loads((shard / "summary.json").read_text())
        if (
            summary["status"] != "passed"
            or summary["feature_dimension"] != 4096
            or summary["projection_applied"]
            or summary["labels_read"]
        ):
            raise ValueError("shard failed frozen-feature contract")
        if signature is None:
            signature = summary["base_signature"]
        elif signature != summary["base_signature"]:
            raise ValueError("shards use different backbone versions or prompts")
        for line in (shard / "manifest.jsonl").open():
            record = json.loads(line)
            key = record["workflow_id"], record["evidence_id"], record["view_id"]
            if key not in planned or key in records:
                raise ValueError("unknown or duplicated encoded source view")
            if planned[key]["source_sha256"] != record["source_sha256"]:
                raise ValueError("encoded raw-source version mismatch")
            if planned[key]["view_count"] != record["view_count"]:
                raise ValueError("encoded material view count differs from plan")
            records[key] = record
    if set(records) != set(planned):
        raise ValueError("feature coverage incomplete; no partial file pooling allowed")
    groups = defaultdict(list)
    for key, record in records.items():
        if (
            planned[key]["split"] != "train"
            or planned[key]["base_revision"] != signature["revision"]
        ):
            raise ValueError("feature split or revision mismatch")
        groups[key[:2]].append(record)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for (workflow, evidence), group in sorted(groups.items()):
        if any(
            Path(identifier).name != identifier or identifier in {".", ".."}
            for identifier in (workflow, evidence)
        ):
            raise ValueError("invalid material identifier")
        if len({record["source_sha256"] for record in group}) != 1:
            raise ValueError("views of one material use different source versions")
        if any(record["view_count"] != len(group) for record in group):
            raise ValueError("original material view count mismatch")
        vectors = []
        for record in group:
            path = Path(record["path"]).resolve()
            if not path.is_relative_to(args.encoded_root.resolve()):
                raise ValueError("feature path escapes encoded root")
            stored = torch.load(path, map_location="cpu", weights_only=True)
            if (
                stored["signature"] != signature
                or stored["cache_key"] != record["cache_key"]
            ):
                raise ValueError("feature tensor provenance mismatch")
            vector = stored["vector"]
            if vector.shape != (4096,) or not torch.isfinite(vector).all():
                raise ValueError("invalid raw frozen feature")
            vectors.append(vector.float())
        pooled = torch.stack(vectors).mean(dim=0)
        destination = args.output_dir / workflow / f"{evidence}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "raw_evidence": pooled,
                "signature": signature,
                "source_sha256": group[0]["source_sha256"],
                "pooling": "equal_mean_all_independent_views",
                "view_ids": sorted(record["view_id"] for record in group),
            },
            destination,
        )
        manifest.append(
            {
                "workflow_id": workflow,
                "evidence_id": evidence,
                "view_count": len(group),
                "path": destination.as_posix(),
                "source_sha256": group[0]["source_sha256"],
            }
        )
    (args.output_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in manifest)
    )
    report = {
        "status": "passed",
        "workflow_count": len({key[0] for key in groups}),
        "original_material_count": len(groups),
        "view_count": len(records),
        "raw_dimension": 4096,
        "projection_applied": False,
        "labels_read": False,
        "signature": signature,
        "pooling": "equal_mean_all_independent_views",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
