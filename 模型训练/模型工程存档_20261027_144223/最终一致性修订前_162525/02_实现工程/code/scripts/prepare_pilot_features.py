"""Plan complete train-only pilot evidence encodings before loading GPU weights.

These views are for independent file-feature pooling, not policy-table lookup.
The original row/cell/segment index remains required for operand retrieval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import yaml
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from audit_engine.readers.chunking import text_views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/smoke5090.yaml"))
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("models/Qwen3-VL-8B-Instruct")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/features_pilot"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    selected = [
        json.loads(line)["workflow_id"]
        for line in Path("data/pilot/pilot20_selection.jsonl").open()
    ]
    # Only the train-input file is read; no target, validation or test file.
    training = {
        x["workflow_id"]: x
        for x in (
            json.loads(line)
            for line in Path("data/normalized_full/train_inputs.jsonl").open()
        )
    }
    if any(identifier not in training for identifier in selected):
        raise ValueError("pilot selection contains a workflow outside train")
    root = Path(config["project"]["data_root"]).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with (args.output_dir / "encoding_jobs.jsonl").open("w") as stream:
        for identifier in selected:
            bundle = training[identifier]
            for evidence in bundle["evidence"]:
                source = (root / evidence["raw_ref"]).resolve()
                if not source.is_relative_to(root) or not source.is_file():
                    raise ValueError("pilot source missing or outside raw data root")
                counts["original_evidence"] += 1
                if evidence["type"] == "image":
                    views = [
                        {
                            "kind": "original_image",
                            "text": "Read this material and preserve visible facts without making an audit decision.",
                        }
                    ]
                else:
                    text = evidence.get("text") or ""
                    views = [
                        dict(view, kind="text_view")
                        for view in text_views(
                            text,
                            tokenizer,
                            config["reader"]["text_chunk_tokens"],
                            config["reader"]["chunk_overlap_tokens"],
                        )
                    ]
                    if not views:
                        counts["unread_text_materials"] += 1
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                for index, view in enumerate(views):
                    record = {
                        "workflow_id": identifier,
                        "source_family_id": bundle["source_family_id"],
                        "evidence_id": evidence["evidence_id"],
                        "raw_ref": evidence["raw_ref"],
                        "source_sha256": digest,
                        "parser_version": evidence["parser_version"],
                        "view_id": f"{evidence['evidence_id']}:enc{index + 1:04d}",
                        "view_count": len(views),
                        "type": evidence["type"],
                        **view,
                        "base_revision": config["model"]["revision"],
                        "split": "train",
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts[view["kind"]] += 1
    summary = {
        "workflow_count": len(selected),
        "source_family_count": len({training[w]["source_family_id"] for w in selected}),
        "counts": dict(counts),
        "text_chunk_tokens": config["reader"]["text_chunk_tokens"],
        "overlap_tokens": config["reader"]["chunk_overlap_tokens"],
        "base_revision": config["model"]["revision"],
        "source_coverage_verified": True,
        "coverage_scope": "parsed_text_and_original_image_files",
        "labels_read": False,
        "features_encoded": False,
        "scope": "independent_evidence_features_only_operand_queries_use_original_segments",
    }
    (args.output_dir / "plan_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
