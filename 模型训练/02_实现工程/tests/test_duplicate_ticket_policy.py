import hashlib

import pytest
import torch
from docx import Document
from openpyxl import Workbook

from audit_engine.contracts import AuditCase
from audit_engine.examples.patent_fixture import sample_case
from audit_engine.inference.runner import scalar_policies
from audit_engine.policy.duplicate_tickets import extend_duplicate_checks


def source_case(tmp_path, explicit_rejection=True):
    clause = "Any ticket already on the list will be rejected." if explicit_rejection else "Tickets on the list require manual review."
    lines = ["The Previously Reimbursed Tickets List is maintained by Finance.", clause]
    document = Document()
    for text in lines:
        document.add_paragraph(text)
    policy = tmp_path / "policy.docx"
    document.save(policy)
    workbook = Workbook()
    workbook.active.title = "Paid"
    workbook.active.append(["Ticket No."])
    workbook.active.append(["T001"])
    table = tmp_path / "Previously_Reimbursed_Tickets.xlsx"
    workbook.save(table)
    receipt = tmp_path / "receipt.txt"
    receipt.write_text("Bus Ticket\nTicket No.: T001")
    raw = sample_case().model_dump()
    raw["evidence"][0].update(raw_ref=receipt.name,
        source_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(), text=receipt.read_text(),
        segments=[{"segment_id": "r1", "text": receipt.read_text()}])
    raw["evidence"][1].update(type="docx", raw_ref=policy.name,
        source_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
        text="\n".join(lines), segments=[{"segment_id": "p" + str(i), "text": line} for i, line in enumerate(lines)])
    raw["evidence"].append({"evidence_id": "e004", "type": "xlsx", "raw_ref": table.name,
        "source_sha256": hashlib.sha256(table.read_bytes()).hexdigest(), "read_status": "obtained",
        "segments": [{"segment_id": "head", "text": "Sheet Paid: A1=Ticket No."},
                     {"segment_id": "row2", "text": "Sheet Paid: A2=T001"}]})
    raw["application"]["attachment_ids"].append("e004")
    raw["application"]["receipt_owners"] = [{"owner": "Applicant", "evidence_ids": ["receipt"]}]
    raw["conditions"][0].update(operation=None, roles=["record"],
        policy_refs=[{"evidence_id": "policy", "segment_id": "p0", "quote": lines[0]}])
    raw["operands"] = [{"condition_id": "cap", "role": "record", "evidence_id": "receipt",
        "value": receipt.read_text(), "value_type": "text", "status": "obtained",
        "refs": [{"evidence_id": "receipt", "segment_id": "r1", "quote": receipt.read_text()}],
        "depends_on": ["receipt"]}]
    return AuditCase.model_validate(raw)


def test_explicit_prohibition_compiles_from_original_policy_and_actual_ticket_read(tmp_path):
    original = source_case(tmp_path)
    result, provenance = extend_duplicate_checks(original, tmp_path)
    assert len(result.conditions) == len(original.conditions) + 1
    assert result.policy_coverage == original.policy_coverage
    policy = scalar_policies(result)[-1]
    _, _, comparison, complete = policy.rebuild(torch.ones(len(result.evidence), dtype=torch.bool), 8)
    assert complete.any() and set(comparison[complete.squeeze(-1).bool()].argmax(-1).tolist()) == {1}
    assert result.conditions[-1].consequence.on_failed == "reject_object"
    assert provenance[0]["reference_answers_read"] is False
    assert extend_duplicate_checks(result, tmp_path)[0] == result


def test_unspecified_rejection_is_not_invented(tmp_path):
    original = source_case(tmp_path, explicit_rejection=False)
    result, provenance = extend_duplicate_checks(original, tmp_path)
    assert result == original and not provenance


def test_modified_original_policy_is_rejected(tmp_path):
    original = source_case(tmp_path)
    (tmp_path / "policy.docx").write_bytes(b"modified")
    with pytest.raises(ValueError, match="source changed"):
        extend_duplicate_checks(original, tmp_path)
