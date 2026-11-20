"""Bind source-reviewed reading targets to recorded source-only reader prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit_engine.contracts import AuditCase
from audit_engine.readers.patent_reader import SourceFacts
from audit_engine.training.patent import read_jsonl


def section(prompt, marker):
    return json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1].lstrip())[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader-root", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--cases", type=Path, default=Path("data/三模块交付_20261027/dataset/reader_cases.jsonl")
    )
    parser.add_argument(
        "--partitions", type=Path, default=Path("data/三模块交付_20261027/dataset/partitions.jsonl")
    )
    parser.add_argument(
        "--source-root", type=Path, default=Path("data_raw/ReimburseBench")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--role-focus", action="store_true")
    args = parser.parse_args()
    cases = {
        v["workflow_id"]: AuditCase.model_validate(v) for v in read_jsonl(args.cases)
    }
    partitions = {v["workflow_id"]: v for v in read_jsonl(args.partitions)}
    labels = {(v["workflow_id"], v["evidence_id"]): v for v in read_jsonl(args.labels)}
    rows, seen = [], set()
    for path in sorted(args.reader_root.glob("shard_*/reader_calls/*.json")):
        call = json.loads(path.read_text())
        key = (call["workflow_id"], call["evidence_id"])
        if key not in labels:
            continue
        annotation = labels[key]
        if (
            partitions[key[0]]["split"] != args.split
            or annotation["split"] != args.split
            or annotation.get("reference_answer_read") is not False
            or not annotation.get("reviewer")
        ):
            raise ValueError(
                "reader QA must match the independently inspected fixed partition"
            )
        case = cases[key[0]]
        material = next(e for e in case.evidence if e.evidence_id == key[1])
        source = (args.source_root / material.raw_ref).resolve()
        if (
            not source.is_relative_to(args.source_root.resolve())
            or hashlib.sha256(source.read_bytes()).hexdigest() != material.source_sha256
        ):
            raise ValueError("reader QA source path/hash mismatch")
        if (
            material.source_sha256 != annotation["source_sha256"]
            or call["source_sha256"] != material.source_sha256
        ):
            raise ValueError("source inspection and reader prompt versions differ")
        prompt = call["prompt"]
        if any(
            value in prompt
            for value in (
                "reference_hint",
                "reference_max_legal",
                "teacher_answer",
                "target_F",
            )
        ):
            raise ValueError(
                "reference or component supervision leaked into a reader prompt"
            )
        roles = section(prompt, "\nROLES:\n")
        requested = {v["role"] if isinstance(v, dict) else v for v in roles}
        view = section(prompt, "\nVIEW:\n")
        if view.get("view_mode"):
            # Reviewed labels here are whole-source observations, not tile labels.
            continue
        if view["kind"] == "image" and material.type == "pdf":
            # These labels were reviewed against PDF text segments; rendered
            # pages need separately approved visual observations.
            continue
        segments = {s["segment_id"]: s["text"] for s in view.get("segments", [])}
        facts = []
        for fact in annotation["facts"]:
            if fact["role"] not in requested:
                continue
            if view["kind"] == "text":
                identifier = fact["segment_id"]
                if identifier not in segments:
                    continue
                if "".join(fact["quote"].split()) not in "".join(
                    segments[identifier].split()
                ):
                    raise ValueError(
                        "reviewed reading quote is not in the actual prompt view"
                    )
            elif fact["segment_id"] is not None or fact["quote"] or not fact["note"]:
                continue
            facts.append(fact)
        target = SourceFacts.model_validate(
            {"facts": facts, "issues": []}
        ).model_dump_json()
        digest = hashlib.sha256(
            (prompt + target + material.source_sha256).encode()
        ).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        rows.append(
            {
                "qa_id": digest,
                "workflow_id": key[0],
                "split": args.split,
                "source_family_id": partitions[key[0]]["source_family_id"],
                "evidence_id": key[1],
                "source_sha256": material.source_sha256,
                "raw_ref": material.raw_ref,
                "image_ref": material.raw_ref if view["kind"] == "image" else None,
                "prompt": prompt,
                "response": target,
                "status": "approved",
                "reviewer": annotation["reviewer"],
                "method": annotation["method"],
                "source_reader_record": str(path),
                "source_review_record": annotation["review_record"],
                "reference_answer_read": False,
            }
        )
    if args.role_focus:
        augmented = {}
        for row in rows:
            definitions = {
                role: meaning
                for condition in cases[row["workflow_id"]].conditions
                for role, meaning in condition.role_definitions.items()
            }
            header, rest = row["prompt"].split("\nROLES:\n", 1)
            stripped = rest.lstrip()
            declared, end = json.JSONDecoder().raw_decode(stripped)
            suffix = stripped[end:]
            declared = [item if isinstance(item, dict) else {"role": item} for item in declared]
            declarations = [
                {
                    **item,
                    "definition": definitions.get(
                        item["role"], item["role"].replace("_", " ")
                    ),
                }
                for item in declared
            ]
            queries = (
                [declarations] + [[item] for item in declarations]
                if len(declarations) > 1
                else [declarations]
            )
            facts = json.loads(row["response"])["facts"]
            for query in queries:
                requested = {item["role"] for item in query}
                prompt = (
                    header
                    + "\nROLES:\n"
                    + json.dumps(query, ensure_ascii=False)
                    + suffix
                )
                target = SourceFacts.model_validate(
                    {
                        "facts": [f for f in facts if f["role"] in requested],
                        "issues": [],
                    }
                ).model_dump_json()
                key = hashlib.sha256(
                    (prompt + target + row["source_sha256"]).encode()
                ).hexdigest()
                augmented[key] = {
                    **row,
                    "qa_id": key,
                    "prompt": prompt,
                    "response": target,
                    "query_variant": "source_bound_role_focus",
                    "parent_prompt_sha256": hashlib.sha256(
                        row["prompt"].encode()
                    ).hexdigest(),
                    "role_definition_case_sha256": cases[row["workflow_id"]].digest(),
                }
        rows = list(augmented.values())
    represented = {r["workflow_id"] for r in rows}
    if represented != {key[0] for key in labels}:
        raise ValueError(
            "some source-reviewed workflows have no matching actual reader prompt"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "qa_records": len(rows),
        "workflows": len(represented),
        "split": args.split,
        "official_answers_used": False,
        "qa_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "reviewed_labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
