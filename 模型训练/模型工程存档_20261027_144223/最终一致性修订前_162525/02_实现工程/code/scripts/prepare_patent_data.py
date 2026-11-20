"""Export source-bound runtime cases and a condition review queue without gold answers."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit_engine.contracts import AuditCase
from audit_engine.training.patent import read_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--normalized-root", type=Path, default=Path("data/normalized_full"))
    p.add_argument("--source-root", type=Path, default=Path("data_raw/ReimburseBench"))
    p.add_argument(
        "--assignments", type=Path, default=Path("data/splits/split_assignments.jsonl")
    )
    p.add_argument("--output", type=Path, default=Path("data/patent_full"))
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(
            "choose a new output directory to preserve existing annotations"
        )
    assignments = {v["workflow_id"]: v for v in read_jsonl(args.assignments)}
    args.output.mkdir(parents=True)
    counts, pending = Counter(), []
    root = args.source_root.resolve()
    all_cases = []
    for split in ("train", "validation", "test"):
        with (args.output / f"{split}_cases.jsonl").open(
            "w", encoding="utf-8"
        ) as stream:
            for row in read_jsonl(args.normalized_root / f"{split}_inputs.jsonl"):
                assignment = assignments[row["workflow_id"]]
                if (
                    assignment["split"] != split
                    or row["source_family_id"] != assignment["source_family_id"]
                ):
                    raise ValueError("source grouping differs from the fixed split")
                materials = []
                unresolved = ["policy_conditions_require_source_review"]
                for item in row["evidence"]:
                    path = (root / item["raw_ref"]).resolve()
                    if not path.is_relative_to(root) or not path.is_file():
                        raise ValueError("invalid raw source path")
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    segments = [
                        {"segment_id": s["segment_id"], "text": s["text"]}
                        for s in item.get("segments", [])
                    ]
                    text = item.get("text") or ""
                    if text and not segments:
                        segments = [{"segment_id": "whole_text", "text": text}]
                    status = "obtained" if text else "not_obtained"
                    if status != "obtained":
                        unresolved.append(
                            f"material_reading_required:{item['evidence_id']}"
                        )
                    materials.append(
                        {
                            "evidence_id": item["evidence_id"],
                            "raw_ref": item["raw_ref"],
                            "source_sha256": digest,
                            "type": item["type"],
                            "text": text or None,
                            "segments": segments,
                            "read_status": status,
                        }
                    )
                source = row["input"]
                application = {
                    key: source[key]
                    for key in (
                        "application",
                        "claimed_amount",
                        "currency",
                        "attachment_ids",
                        "receipt_owners",
                    )
                }
                case = AuditCase.model_validate(
                    {
                        "workflow_id": row["workflow_id"],
                        "input_version": "patent_source_bound_v1",
                        "application": application,
                        "evidence": materials,
                        "conditions": [],
                        "operands": [],
                        "amount_units": [
                            {
                                "object_id": "claim",
                                "scope": "application",
                                "claimed_amount": source["claimed_amount"],
                                "currency": source["currency"],
                                "limit_status": "unknown",
                            }
                        ],
                        "policy_coverage": "pending",
                        "unresolved": unresolved,
                    }
                )
                stream.write(case.model_dump_json() + "\n")
                all_cases.append(case)
                counts[split] += 1
                pending.append(
                    {
                        "workflow_id": case.workflow_id,
                        "split": split,
                        "source_family_id": row["source_family_id"],
                        "case_sha256": case.digest(),
                        "status": "pending",
                        "required": [
                            "policy_applicability_and_scope",
                            "conditions_and_consequences",
                            "typed_operands_and_source_quotes",
                            "condition_state",
                            "relation_and_reference_labels",
                            "disjoint_amount_units_and_caps",
                        ],
                        "unresolved": unresolved,
                    }
                )
    with (args.output / "cases.jsonl").open("w", encoding="utf-8") as stream:
        for case in all_cases:
            stream.write(case.model_dump_json() + "\n")
    (args.output / "condition_review_queue.jsonl").write_text(
        "".join(json.dumps(v, ensure_ascii=False) + "\n" for v in pending),
        encoding="utf-8",
    )
    (args.output / "annotations.jsonl").write_text("", encoding="utf-8")
    (args.output / "partitions.jsonl").write_text(
        "".join(
            json.dumps(
                {k: row[k] for k in ("workflow_id", "source_family_id", "split")}
            )
            + "\n"
            for row in pending
        ),
        encoding="utf-8",
    )
    summary = {
        "workflows": dict(counts),
        "approved_condition_annotations": 0,
        "answer_fields_exported": False,
        "status": "source_bound_inputs_ready_condition_review_pending",
    }
    (args.output / "preparation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
