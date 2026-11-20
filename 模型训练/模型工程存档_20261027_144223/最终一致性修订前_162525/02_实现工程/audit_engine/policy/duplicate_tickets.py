"""Compile an explicit duplicate-ticket rejection clause with original sources.

This narrow compiler recognizes a written, unconditional prohibition. It never
approves general policy coverage, infers a rejection from an unspecified rule,
or reads benchmark targets.
"""

import hashlib
import json
import re
from pathlib import Path

from docx import Document

from audit_engine.contracts import AuditCase, Condition, Operand
from audit_engine.readers.table_sets import bind_identifier_roles


def extend_duplicate_checks(case: AuditCase, source_root: Path):
    if any(c.operation and c.operation.name == "NOT_IN_SET" for c in case.conditions):
        return case, []
    clauses = []
    for material in case.evidence:
        if material.type != "docx" or not re.search(r"policy|regulation", material.raw_ref, re.I):
            continue
        if not any("previously reimbursed" in segment.text.casefold() for segment in material.segments):
            continue
        found = [segment for segment in material.segments if re.search(
            r"any\s+ticket\b.*\balready\b.*\blist\b.*\b(?:will be rejected|may not be claimed)",
            segment.text, re.I)]
        if not found:
            continue
        path = (source_root / material.raw_ref).resolve()
        if not path.is_relative_to(source_root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != material.source_sha256:
            raise ValueError("duplicate-ticket policy source changed or is outside root")
        document = Document(path)
        original = [p.text for p in document.paragraphs]
        original.extend(cell.text for table in document.tables for row in table.rows for cell in row.cells)
        for segment in found:
            if not any("".join(segment.text.split()) == "".join(text.split()) for text in original):
                raise ValueError("duplicate rejection clause is not in the original document")
            clauses.append({"evidence_id": material.evidence_id, "segment_id": segment.segment_id,
                            "quote": segment.text})
    tables = [e for e in case.evidence if e.type == "xlsx" and re.search(
        r"previously[_ ]reimbursed.*tickets?", Path(e.raw_ref).name, re.I)]
    if not clauses or len(tables) != 1:
        return case, []
    table = tables[0]
    receipt_ids = {identifier for owner in case.application.receipt_owners
                   for identifier in owner.get("evidence_ids", [])}
    raw, provenance = case.model_dump(), []
    for receipt_id in sorted(receipt_ids):
        if len(case.amount_units) != 1:
            continue  # Disjoint allocations must be established separately.
        readings = []
        for operand in case.operands:
            if operand.evidence_id != receipt_id or operand.status != "obtained" or not isinstance(operand.value, str):
                continue
            matches = re.findall(r"\bTicket\s*(?:No\.?|Number|#)\s*[:#]?\s*([A-Za-z0-9][A-Za-z0-9._/-]*)",
                                 operand.value, re.I)
            readings.extend((value, operand) for value in matches)
        if len({value for value, _ in readings}) != 1:
            continue
        identifier = "ticket_not_previously_reimbursed_" + receipt_id
        actual_role = "actual_" + receipt_id + "_ticket_number"
        previous_role = "previously_reimbursed_" + table.evidence_id + "_ticket_numbers"
        condition = Condition.model_validate({"condition_id": identifier,
            "object_id": case.amount_units[0].object_id,
            "question": "Is the original ticket absent from the Finance previously reimbursed ticket list?",
            "kind": "required", "roles": [actual_role, previous_role],
            "role_origins": {actual_role: "receipt", previous_role: "background"},
            "role_types": {actual_role: "text", previous_role: "set"},
            "role_definitions": {actual_role: "Read the literal Ticket No. printed on this original ticket.",
                previous_role: "Read the complete Ticket No. identifier column of the original Finance reimbursement register."},
            "policy_refs": clauses,
            "operation": {"name": "NOT_IN_SET", "left_role": actual_role, "right_role": previous_role,
                          "value_type": "text", "text_normalization": "casefold_alnum"},
            "scope": "item", "consequence": {"on_failed": "reject_object", "on_gap": "request_material"},
            "compile_status": "verified", "verification_method": "original_explicit_duplicate_rejection_clause_v1"})
        raw["conditions"].append(condition.model_dump())
        value, parent = readings[0]
        observation = Operand.model_validate({"condition_id": identifier, "role": actual_role,
            "evidence_id": receipt_id, "value": value, "value_type": "text", "status": "obtained",
            "refs": [ref.model_dump() for ref in parent.refs], "depends_on": list(parent.depends_on),
            "dependency_groups": parent.dependency_groups})
        raw["operands"].append(observation.model_dump())
        provenance.append({"condition_id": identifier, "method": "literal_ticket_number_from_actual_reader_record",
                           "receipt_id": receipt_id, "policy_refs": clauses,
                           "parent_read": parent.model_dump(), "reference_answers_read": False})
    if not provenance:
        return case, []
    raw["input_version"] = "patent_reader_v1:explicit_duplicate_clause:" + hashlib.sha256(
        json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    result, columns = bind_identifier_roles(AuditCase.model_validate(raw), source_root)
    # A recognized clause without its full source register remains a candidate.
    needed = {record["condition_id"] for record in provenance}
    obtained = {o.condition_id for o in result.operands if o.role.startswith("previously_reimbursed_")
                and o.status == "obtained"}
    if not needed.issubset(obtained):
        return case, [{"status": "unresolved", "reason": "complete_register_not_read"}]
    return result, provenance + columns
