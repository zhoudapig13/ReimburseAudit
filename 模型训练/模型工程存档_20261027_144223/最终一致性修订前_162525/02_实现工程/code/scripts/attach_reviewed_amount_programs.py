"""Attach deterministic source money programs without changing encoder inputs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from audit_engine.contracts import AuditCase, ConditionAnnotation
from audit_engine.training.patent import feature_path, load_features, read_jsonl, validate_annotations


def unchanged_encoder_inputs(old, new):
    def policies(case):
        values = [condition.model_dump() for condition in case.conditions]
        for value in values:
            value.pop("verification_method")
        return values

    return (old.application == new.application and old.evidence == new.evidence
            and policies(old) == policies(new))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "features").mkdir()
    compiled = {row["workflow_id"]: AuditCase.model_validate(row)
                for row in read_jsonl(Path("data/patent_reviewed/policy_cases_v7.jsonl"))}
    old_cases = [AuditCase.model_validate(row) for row in read_jsonl(args.input_run / "reader_cases.jsonl")]
    cases, checks = [], []
    for old in old_cases:
        parent = old.workflow_id.split("_drop_")[0]
        reviewed = compiled[parent]
        raw = old.model_dump()
        if reviewed.amount_programs:
            raw["policy_numbers"] = [number.model_dump() for number in reviewed.policy_numbers]
            raw["amount_programs"] = [program.model_dump() for program in reviewed.amount_programs]
            raw["conditions"] = [condition.model_dump() for condition in reviewed.conditions]
        new = AuditCase.model_validate(raw)
        if not unchanged_encoder_inputs(old, new):
            raise ValueError("changed neural encoder inputs require fresh backbone encoding")
        features = load_features(feature_path(args.input_run / "features", old.workflow_id))
        features.validate(old, 4096)
        features.case_sha256 = new.digest()
        features.validate(new, 4096)
        torch.save(asdict(features), feature_path(args.output / "features", new.workflow_id))
        cases.append(new)
        checks.append({"workflow_id": new.workflow_id, "old_case_sha256": old.digest(),
                       "new_case_sha256": new.digest(), "application_evidence_and_policy_encoding_inputs_identical": True,
                       "amount_programs": len(new.amount_programs)})
    hashes = {case.workflow_id: case.digest() for case in cases}
    annotations = []
    for row in read_jsonl(args.input_run / "annotations.jsonl"):
        row["case_sha256"] = hashes[row["workflow_id"]]
        annotations.append(ConditionAnnotation.model_validate(row))
    partitions = {row["workflow_id"]: row for row in read_jsonl(Path("data/patent_verified_deletions_v1/partitions.jsonl"))}
    validate_annotations(cases, annotations, partitions)
    (args.output / "reader_cases.jsonl").write_text("".join(case.model_dump_json() + "\n" for case in cases))
    (args.output / "annotations.jsonl").write_text("".join(annotation.model_dump_json() + "\n" for annotation in annotations))
    (args.output / "feature_binding_verification.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    print(json.dumps({"workflows": len(cases), "with_source_amount_programs": sum(bool(case.amount_programs) for case in cases),
                      "encoder_inputs_unchanged_and_verified": True}), flush=True)


if __name__ == "__main__":
    main()
