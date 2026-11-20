import hashlib
import importlib.util
from pathlib import Path

import pytest
from openpyxl import Workbook

from audit_engine.contracts import AuditCase
from audit_engine.examples.patent_fixture import sample_case

SPEC = importlib.util.spec_from_file_location(
    "bind_identifiers", Path(__file__).parents[1] / "code/scripts/bind_original_identifier_columns.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def case_with_table(tmp_path, duplicate_header=False):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Paid"
    sheet.append(["Ticket No.", "Passenger"])
    sheet.append(["T001", "Person A"])
    sheet.append(["T002", "Person B"])
    if duplicate_header:
        workbook.create_sheet("Other").append(["Ticket Number"])
    path = tmp_path / "tickets.xlsx"
    workbook.save(path)
    raw = sample_case().model_dump()
    raw["evidence"].append({"evidence_id": "e004", "type": "xlsx", "raw_ref": path.name,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "read_status": "obtained", "segments": [
            {"segment_id": "head", "text": "Sheet Paid: A1=Ticket No. | B1=Passenger"},
            {"segment_id": "row2", "text": "Sheet Paid: A2=T001 | B2=Person A"},
            {"segment_id": "row3", "text": "Sheet Paid: A3=T002 | B3=Person B"}]})
    raw["application"]["attachment_ids"].append("e004")
    raw["conditions"][0].update(roles=["ticket", "previously_reimbursed_e004_ticket_numbers"],
        operation={"name": "NOT_IN_SET", "left_role": "ticket",
                   "right_role": "previously_reimbursed_e004_ticket_numbers", "value_type": "text"})
    raw["operands"] = []
    return AuditCase.model_validate(raw)


def test_complete_original_column_is_bound_without_approving_coverage(tmp_path):
    case = case_with_table(tmp_path)
    result, provenance = MODULE.bind_identifier_roles(case, tmp_path)
    assert result.operands[0].value == ["T001", "T002"]
    assert {r.segment_id for r in result.operands[0].refs} == {"row2", "row3"}
    assert result.policy_coverage == case.policy_coverage
    assert provenance[0]["entire_column_read"]
    again, records = MODULE.bind_identifier_roles(result, tmp_path)
    assert again == result and not records


def test_ambiguous_workbook_headers_remain_unresolved(tmp_path):
    case = case_with_table(tmp_path, duplicate_header=True)
    result, provenance = MODULE.bind_identifier_roles(case, tmp_path)
    assert not result.operands
    assert provenance[0]["status"] == "unresolved"


def test_changed_original_workbook_is_rejected(tmp_path):
    case = case_with_table(tmp_path)
    (tmp_path / "tickets.xlsx").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash changed"):
        MODULE.bind_identifier_roles(case, tmp_path)
