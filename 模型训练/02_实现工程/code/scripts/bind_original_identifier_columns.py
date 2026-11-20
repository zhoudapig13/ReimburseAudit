"""Bind explicitly declared ticket-set roles to complete original XLSX columns.

This reader does not infer policy outcomes or approve policy coverage. A role
must identify its source file, and that workbook must contain exactly one
unambiguous ticket identifier column. Existing contradictory obtained readings
are retained; only absent readings are filled.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit_engine.contracts import AuditCase
from audit_engine.readers.table_sets import bind_identifier_roles
from audit_engine.training.patent import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("data_raw/ReimburseBench"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        for row in read_jsonl(args.cases):
            case, provenance = bind_identifier_roles(AuditCase.model_validate(row), args.source_root)
            stream.write(case.model_dump_json() + "\n")
            records.extend(provenance)
    args.provenance.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"bindings": sum(r["status"] == "bound" for r in records),
                      "policy_coverage_changed": False, "reference_answers_read": False}))


if __name__ == "__main__":
    main()
