"""Apply explicit original-table readers and exact source-receipt allocations."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit_engine.contracts import AuditCase
from audit_engine.policy.allocations import allocate_receipt_units
from audit_engine.readers.table_sets import read_identifier_column
from audit_engine.readers.text_fields import read_original_text_field
from audit_engine.inference.runner import scalar_policies
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("data_raw/ReimburseBench"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()
    bindings = json.loads(args.bindings.read_text())["workflows"]
    cases, records, applied = [], [], set()
    for line in args.cases.read_text().splitlines():
        case = AuditCase.model_validate_json(line)
        if case.workflow_id in bindings:
            binding = bindings[case.workflow_id]
            if binding["source_case_sha256"] != case.digest():
                raise ValueError("source binding belongs to a different reader input version")
            source_digest = case.digest()
            readers = []
            for locator in binding.get("text_fields", []):
                observation, provenance = read_original_text_field(case, args.source_root, **locator)
                prior = [o for o in case.operands if o.role == locator["role"]
                         and o.condition_id == locator["condition_id"] and o.evidence_id == locator["evidence_id"]]
                raw = case.model_dump()
                raw["operands"] = [o.model_dump() for o in case.operands if o not in prior] + [observation.model_dump()]
                raw["input_version"] = "patent_reader_v1:original_text_field:" + hashlib.sha256(
                    (case.digest() + json.dumps(provenance, sort_keys=True)).encode()).hexdigest()
                case = AuditCase.model_validate(raw)
                provenance["prior_observations"] = [o.model_dump() for o in prior]
                readers.append(provenance)
            for locator in binding.get("table_sets", []):
                observations, provenance = read_identifier_column(case, args.source_root, **locator)
                raw = case.model_dump()
                prior = [o for o in case.operands if o.role == locator["role"]
                         and o.condition_id in locator["condition_ids"]]
                raw["operands"] = [o.model_dump() for o in case.operands if o not in prior]
                raw["operands"].extend(o.model_dump() for o in observations)
                raw["input_version"] = "patent_reader_v1:source_column:" + hashlib.sha256(
                    (case.digest() + json.dumps(provenance, sort_keys=True)).encode()).hexdigest()
                case = AuditCase.model_validate(raw)
                provenance["prior_observations"] = [o.model_dump() for o in prior]
                readers.append(provenance)
            allocation = None
            if binding.get("receipt_allocations"):
                case, allocation = allocate_receipt_units(case, binding["receipt_allocations"])
            if binding.get("text_fields") and case.policy_coverage == "verified" and "material_reader_failure" in case.unresolved:
                active = torch.ones(len(case.evidence), dtype=torch.bool)
                complete = True
                for policy in scalar_policies(case):
                    roles, required, _, comparisons = policy.rebuild(active, 8)
                    complete &= bool((roles.any(-1) | ~required).all())
                    complete &= (bool((comparisons.squeeze(-1).bool() | ~roles.any(0)).all())
                                 if policy.comparison_required else policy.semantic_operands_ready(active))
                if complete:
                    raw = case.model_dump()
                    raw["unresolved"].remove("material_reader_failure")
                    case = AuditCase.model_validate(raw)
                    readers.append({"resolved_flag": "material_reader_failure",
                        "reason": "All required roles and deterministic comparisons in the verified policy scope now have unambiguous source-bound inputs. Original failed reads remain in provenance."})
            records.append({"workflow_id": case.workflow_id, "source_case_sha256": source_digest,
                            "output_case_sha256": case.digest(), "table_readers": readers,
                            "allocation": allocation})
            applied.add(case.workflow_id)
        cases.append(case)
    if applied != set(bindings):
        raise ValueError("source binding contains workflows absent from the input")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text("".join(case.model_dump_json() + "\n" for case in cases))
    temporary.replace(args.output)
    args.provenance.write_text(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
