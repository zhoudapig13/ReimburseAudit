"""Compile recorded source inspections into P; never insert supervised readings."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit_engine.contracts import AuditCase, Condition
from audit_engine.training.patent import read_jsonl


def compile_case(case: AuditCase, review: dict) -> AuditCase:
    if (
        review["workflow_id"] != case.workflow_id
        or review.get("runtime_operands_injected") is not False
    ):
        raise ValueError(
            "source review is not a policy-only inspection of this workflow"
        )
    expected = {e.evidence_id: e.source_sha256 for e in case.evidence}
    if (
        review["source_sha256"] != expected
        or not review.get("reviewer")
        or not review.get("reviewed_at")
    ):
        raise ValueError(
            "policy inspection source versions or reviewer record are invalid"
        )
    policy_keys = {"id", "question", "policy_refs", "source", "operation", "roles", "right_constant", "currency", "value_type",
                   "scope", "text_normalization", "role_origins", "role_definitions", "role_types", "derived_roles", "on_failed", "on_gap", "right_role"}
    policy_version = {key: review.get(key) for key in ("workflow_id", "source_sha256", "method", "reviewer", "reviewed_at",
                      "policy_numbers", "amount_programs", "policy_coverage", "coverage_review")}
    policy_version["confirmed_policy"] = [{key: value for key, value in rule.items() if key in policy_keys} for rule in review["confirmed_policy"]]
    digest = hashlib.sha256(json.dumps(policy_version, sort_keys=True).encode()).hexdigest()
    segments = {
        s.segment_id: (e.evidence_id, s.text) for e in case.evidence for s in e.segments
    }
    compiled = []
    for rule in review["confirmed_policy"]:
        refs = []
        if "policy_refs" in rule:
            for identifier in rule["policy_refs"]:
                evidence_id, quote = segments[identifier]
                refs.append(
                    {
                        "evidence_id": evidence_id,
                        "segment_id": identifier,
                        "quote": quote,
                    }
                )
        else:
            source = rule["source"]
            refs = source if isinstance(source, list) else [source]
        value_type = rule.get(
            "value_type",
            "money"
            if rule.get("currency")
            else "date"
            if rule["operation"] == "WITHIN"
            else "decimal"
            if rule["operation"] in {"LE", "GE"}
            else "text",
        )
        constant = rule.get("right_constant")
        roles = rule["roles"]
        if rule["operation"] is not None and constant is None and len(roles) < 2:
            raise ValueError(
                "reviewed comparison requires an explicit constant or two declared roles"
            )
        question = rule.get("question", rule["id"].replace("_", " ") + "?")
        if rule.get("scope"):
            question += " Scope of this check: " + rule["scope"] + "."
        condition = Condition.model_validate(
            {
                "condition_id": rule["id"],
                "object_id": "claim",
                "question": question,
                "kind": "open_text" if rule["operation"] is None else "amount" if value_type == "money" else "required",
                "roles": roles,
                "role_origins": rule.get("role_origins", {}),
                "role_definitions": rule.get("role_definitions", {}),
                "role_types": rule.get("role_types", {}),
                "derived_roles": rule.get("derived_roles", {}),
                "policy_refs": refs,
                "operation": None if rule["operation"] is None else {
                    "name": rule["operation"],
                    "left_role": roles[0],
                    "right_role": None if constant is not None else rule.get("right_role", roles[1]),
                    "right_constant": constant,
                    "right_from_policy": constant is not None,
                    "value_type": value_type,
                    "currency": rule.get("currency"),
                    "text_normalization": rule.get("text_normalization", "exact"),
                },
                "scope": "application",
                "consequence": {
                    "on_failed": rule.get("on_failed", "manual_review"),
                    "on_gap": rule.get("on_gap", "request_material"),
                    "on_contradictory": "manual_review",
                },
                "compile_status": "verified",
                "verification_method": review["method"] + ":" + digest,
            }
        )
        compiled.append(condition.model_dump())
    raw = case.model_dump()
    numbers = []
    for parameter in review.get("policy_numbers", []):
        parameter = dict(parameter)
        identifiers = parameter.pop("policy_refs", None)
        if identifiers is not None:
            parameter["refs"] = [{"evidence_id": segments[key][0], "segment_id": key, "quote": segments[key][1]} for key in identifiers]
        parameter.update(compile_status="verified", verification_method=review["method"] + ":" + digest)
        numbers.append(parameter)
    programs = []
    for program in review.get("amount_programs", []):
        program = dict(program)
        identifiers = program.pop("unlimited_policy_refs", None)
        if identifiers is not None:
            program["unlimited_refs"] = [{"evidence_id": segments[key][0], "segment_id": key, "quote": segments[key][1]} for key in identifiers]
        program.update(compile_status="verified", verification_method=review["method"] + ":" + digest)
        programs.append(program)
    coverage = review.get("policy_coverage", "pending")
    if coverage == "verified" and (not review.get("coverage_review", {}).get("method") or not review.get("coverage_review", {}).get("reviewer")):
        raise ValueError("complete policy coverage requires a separate source-scope review")
    raw.update(conditions=compiled, operands=[], policy_coverage=coverage, policy_numbers=numbers, amount_programs=programs)
    raw["unresolved"] = sorted(
        set(
            [
                s
                for s in case.unresolved
                if s not in {"policy_conditions_require_source_review", "remaining_policy_coverage_requires_review"}
            ]
            + ([] if coverage == "verified" else ["remaining_policy_coverage_requires_review"])
        )
    )
    return AuditCase.model_validate(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=Path("data/三模块交付_20261027/dataset/reader_cases.jsonl")
    )
    parser.add_argument(
        "--reviews", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [AuditCase.model_validate(row) for row in read_jsonl(args.cases)]
    count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for case in cases:
            review = args.reviews / (case.workflow_id + ".json")
            if review.is_file():
                case = compile_case(case, json.loads(review.read_text()))
                count += 1
            stream.write(case.model_dump_json() + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "source_reviewed_workflows": count,
                "case_count": len(cases),
                "output": str(args.output),
                "supervised_values_inserted": False,
            }
        )
    )


if __name__ == "__main__":
    main()
