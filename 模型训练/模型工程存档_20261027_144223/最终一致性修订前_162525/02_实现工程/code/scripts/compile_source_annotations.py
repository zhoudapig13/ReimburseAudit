"""Bind individually inspected supervision to immutable actual-reader cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit_engine.contracts import AuditCase, ConditionAnnotation
from audit_engine.training.patent import read_jsonl, validate_annotations


def compile_annotations(case, review, partition):
    if not case.input_version.startswith("patent_reader_v1:"):
        raise ValueError("supervision requires actual reader cases")
    if review["source_sha256"] != {
        e.evidence_id: e.source_sha256 for e in case.evidence
    }:
        raise ValueError("source inspection version changed")
    if not review.get("reviewer") or not review.get("reviewed_at"):
        raise ValueError("source supervision needs a recorded reviewer and time")
    if (
        partition["split"] != "train"
        and review.get("reference_answer_read") is not False
    ):
        raise ValueError(
            "validation source review must be independent of reference answers"
        )
    rules = {r["id"]: r for r in review["confirmed_policy"]}
    digest = hashlib.sha256(json.dumps(review, sort_keys=True).encode()).hexdigest()
    output = []
    known = {e.evidence_id for e in case.evidence}
    for condition in case.conditions:
        rule = rules[condition.condition_id]
        support, conflict = (
            set(rule.get("positive_sources", [])),
            set(rule.get("conflict_sources", [])),
        )
        unrelated = set(rule.get("negative_sources", []))
        if not (support | conflict | unrelated).issubset(known):
            raise ValueError("source label has unknown evidence")
        if support & conflict or unrelated & (support | conflict):
            raise ValueError(
                "overlapping inspected polarities require explicit separate review"
            )
        state = rule.get("candidate_state")
        if state is None and not (support | conflict | unrelated):
            continue
        relations, references = [], []
        for identifier in sorted(support | conflict | unrelated):
            relation = (
                "support"
                if identifier in support
                else "conflict"
                if identifier in conflict
                else "unknown"
            )
            relations.append(
                {"evidence_id": identifier, "relation": relation, "approved": True}
            )
            for polarity, positives in (("support", support), ("conflict", conflict)):
                references.append(
                    {
                        "evidence_id": identifier,
                        "polarity": polarity,
                        "label": int(identifier in positives),
                        "approved": True,
                    }
                )
        output.append(
            ConditionAnnotation.model_validate(
                {
                    "workflow_id": case.workflow_id,
                    "condition_id": condition.condition_id,
                    "split": partition["split"],
                    "source_family_id": partition["source_family_id"],
                    "case_sha256": case.digest(),
                    "status": "approved",
                    "method": review["method"] + ":" + digest,
                    "reviewer": review["reviewer"],
                    "review_reason": "Individually inspected policy and source observations recorded in "
                    + case.workflow_id
                    + ".json; only explicitly reviewed source polarities are labelled; uninspected sources remain masked.",
                    "state": state,
                    "relations": relations,
                    "references": references,
                }
            )
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument(
        "--reviews", type=Path, default=Path("data/patent_source_review")
    )
    parser.add_argument(
        "--partitions", type=Path, default=Path("data/patent_full/partitions.jsonl")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [AuditCase.model_validate(v) for v in read_jsonl(args.cases)]
    partitions = {v["workflow_id"]: v for v in read_jsonl(args.partitions)}
    annotations = []
    for case in cases:
        path = args.reviews / (case.workflow_id + ".json")
        if path.is_file():
            annotations.extend(
                compile_annotations(
                    case, json.loads(path.read_text()), partitions[case.workflow_id]
                )
            )
    validate_annotations(cases, annotations, partitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(
        "".join(a.model_dump_json() + "\n" for a in annotations), encoding="utf-8"
    )
    tmp.replace(args.output)
    print(
        json.dumps(
            {
                "annotations": len(annotations),
                "workflows": len({a.workflow_id for a in annotations}),
                "states": sum(a.state is not None for a in annotations),
                "runtime_values_inserted": False,
            }
        )
    )


if __name__ == "__main__":
    main()
