"""Read complete identifier sets from declared columns of original XLSX files."""

import hashlib
import json
import re
from pathlib import Path

from openpyxl import load_workbook

from audit_engine.contracts import AuditCase, Operand


def read_identifier_column(case: AuditCase, source_root: Path, *, evidence_id: str,
                           sheet: str, header: str, role: str, condition_ids: list[str]):
    material = next(e for e in case.evidence if e.evidence_id == evidence_id)
    path = (source_root / material.raw_ref).resolve()
    if not path.is_relative_to(source_root.resolve()) or material.type != "xlsx":
        raise ValueError("identifier column requires an original XLSX inside the source root")
    if hashlib.sha256(path.read_bytes()).hexdigest() != material.source_sha256:
        raise ValueError("source workbook hash changed")
    conditions = {c.condition_id: c for c in case.conditions}
    for key in condition_ids:
        condition = conditions[key]
        operation = condition.operation
        is_set = condition.role_types.get(role) == "set" or (
            operation and operation.right_role == role
            and operation.name in {"IN_SET", "NOT_IN_SET", "AUTHORIZED"})
        if role not in condition.roles or not is_set or role in condition.derived_roles:
            raise ValueError("column binding must target a declared primitive set role")
    if not condition_ids or len(set(condition_ids)) != len(condition_ids):
        raise ValueError("column binding needs distinct target conditions")
    workbook = load_workbook(path, read_only=True, data_only=False)
    try:
        table = workbook[sheet]
        def normalize(text):
            return re.sub(r"[^a-z0-9]", "", str(text).casefold())
        headers = [cell for row in table.iter_rows(min_row=1, max_row=min(table.max_row, 20))
                   for cell in row if cell.value is not None and normalize(cell.value) == normalize(header)]
        if len(headers) != 1:
            raise ValueError("column header is absent or ambiguous")
        start = headers[0]
        values, refs = [], {}
        for row in table.iter_rows(min_row=start.row + 1, min_col=start.column, max_col=start.column):
            cell = row[0]
            if cell.value is None or str(cell.value).strip() == "":
                continue
            if cell.data_type == "f" or not isinstance(cell.value, str):
                raise ValueError("identifier column contains a formula or non-text identifier")
            value = cell.value
            marker = cell.coordinate + "=" + value
            segments = [s for s in material.segments if s.text.startswith("Sheet " + sheet + ":") and marker in s.text]
            if len(segments) != 1:
                raise ValueError("source cell cannot be uniquely bound to its parsed row")
            segment = segments[0]
            refs[segment.segment_id] = {"evidence_id": evidence_id, "segment_id": segment.segment_id,
                                        "quote": segment.text, "kind": "text"}
            values.append(value)
        if not refs:
            marker = start.coordinate + "=" + str(start.value)
            segments = [s for s in material.segments if s.text.startswith("Sheet " + sheet + ":") and marker in s.text]
            if len(segments) != 1:
                raise ValueError("empty identifier table needs its source header")
            refs[segments[0].segment_id] = {"evidence_id": evidence_id,
                "segment_id": segments[0].segment_id, "quote": segments[0].text, "kind": "text"}
    finally:
        workbook.close()
    observations = [Operand.model_validate({"condition_id": key, "role": role,
        "evidence_id": evidence_id, "value": sorted(set(values)), "value_type": "set",
        "status": "obtained", "refs": list(refs.values()), "depends_on": [evidence_id]})
        for key in condition_ids]
    for observation in observations:
        case.model_copy(update={"operands": [observation]}).check_links()
    return observations, {"reader": "original_xlsx_identifier_column_v1", "evidence_id": evidence_id,
        "source_sha256": material.source_sha256, "sheet": sheet, "header": header,
        "role": role, "condition_ids": condition_ids, "data_rows": len(values),
        "distinct_members": len(set(values)), "entire_column_read": True}


def bind_identifier_roles(case, source_root):
    raw = case.model_dump()
    provenance = []
    requests = {}
    for condition in case.conditions:
        operation = condition.operation
        if not operation or operation.name not in {"IN_SET", "NOT_IN_SET", "AUTHORIZED"}:
            continue
        role = operation.right_role
        if not role or "ticket" not in role.casefold():
            continue
        if any(o.condition_id == condition.condition_id and o.role == role
               and o.status == "obtained" for o in case.operands):
            continue
        matches = re.findall(r"(?:^|_)(e\d+)(?=_|$)", role)
        if len(matches) != 1:
            continue
        requests.setdefault((matches[0], role), []).append(condition.condition_id)
    for (evidence_id, role), condition_ids in requests.items():
        material = next((e for e in case.evidence if e.evidence_id == evidence_id), None)
        if material is None or material.type != "xlsx":
            continue
        source = (source_root / material.raw_ref).resolve()
        if not source.is_relative_to(source_root.resolve()):
            raise ValueError("source workbook outside the dataset root")
        if hashlib.sha256(source.read_bytes()).hexdigest() != material.source_sha256:
            raise ValueError("source workbook hash changed")
        workbook = load_workbook(source, read_only=True, data_only=False)
        candidates = []
        try:
            for sheet in workbook:
                for row in sheet.iter_rows(min_row=1, max_row=min(sheet.max_row, 20)):
                    for cell in row:
                        if isinstance(cell.value, str) and re.sub(
                            r"[^a-z0-9]", "", cell.value.casefold()
                        ) in {"ticketno", "ticketnumber", "ticketid"}:
                            candidates.append((sheet.title, cell.value))
        finally:
            workbook.close()
        if len(candidates) != 1:
            provenance.append({"evidence_id": evidence_id, "role": role,
                               "status": "unresolved", "reason": "ambiguous_or_missing_header"})
            continue
        sheet, header = candidates[0]
        observations, record = read_identifier_column(
            case, source_root, evidence_id=evidence_id, sheet=sheet,
            header=header, role=role, condition_ids=condition_ids,
        )
        raw["operands"].extend(o.model_dump() for o in observations)
        record.update(workflow_id=case.workflow_id, status="bound")
        provenance.append(record)
    if any(p["status"] == "bound" for p in provenance):
        digest = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        raw["input_version"] = case.input_version + ":original_identifier_columns:" + digest
    return AuditCase.model_validate(raw), provenance


