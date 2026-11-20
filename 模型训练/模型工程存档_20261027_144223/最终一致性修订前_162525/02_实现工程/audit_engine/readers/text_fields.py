"""Read uniquely located fields from original DOCX, without predicted values."""

import hashlib
import re
from pathlib import Path

from docx import Document

from audit_engine.contracts import Operand
from audit_engine.readers.value_formats import date_value


def read_original_text_field(case, source_root: Path, *, evidence_id, condition_id,
                             role, mode, label=None, first_comma_part=False,
                             start_contains=None, end_contains=None):
    material = next(e for e in case.evidence if e.evidence_id == evidence_id)
    condition = next(c for c in case.conditions if c.condition_id == condition_id)
    if role not in condition.roles or role in condition.derived_roles:
        raise ValueError("field reader needs a declared primitive role")
    source = (source_root / material.raw_ref).resolve()
    if not source.is_relative_to(source_root.resolve()) or material.type != "docx":
        raise ValueError("field reader requires an original DOCX inside the source root")
    if hashlib.sha256(source.read_bytes()).hexdigest() != material.source_sha256:
        raise ValueError("original text source hash changed")
    document = Document(source)
    paragraphs = [p.text for p in document.paragraphs]
    paragraphs += [cell.text for table in document.tables for row in table.rows for cell in row.cells]
    # parser_v1 serializes a DOCX data row as header/value pairs. Verify that
    # exact representation against the original cells, rather than expecting
    # the combined row to exist as a single paragraph.
    for table in document.tables:
        rows = [[" ".join(cell.text.split()) for cell in row.cells] for row in table.rows]
        headers = rows[0] if rows else []
        for cells in rows:
            paragraphs.append(" | ".join(
                f"{headers[index]}: {value}" if headers[index] and headers[index] != value else value
                for index, value in enumerate(cells) if value))
    def normalize(text):
        return "".join(text.split())

    def locate(predicate):
        matches = [s for s in material.segments if predicate(s.text)]
        if len(matches) != 1:
            raise ValueError("source field locator is absent or ambiguous")
        segment = matches[0]
        if not any(normalize(segment.text) == normalize(text) for text in paragraphs):
            raise ValueError("parsed field is not present in the original DOCX")
        return segment

    if mode == "labeled_text":
        if not label or start_contains or end_contains:
            raise ValueError("labeled text needs only an explicit field label")
        pattern = re.compile(r"^\s*" + re.escape(label) + r"\s*[:：]\s*(.+)$", re.IGNORECASE)
        segment = locate(lambda text: bool(pattern.fullmatch(text)))
        value = pattern.fullmatch(segment.text).group(1).strip()
        if first_comma_part:
            value = value.split(",", 1)[0].strip()
        if not value:
            raise ValueError("source text field is empty")
        segments, value_type = [segment], "text"
    elif mode == "date_interval":
        if label or first_comma_part or not start_contains or not end_contains:
            raise ValueError("date interval needs two distinct paragraph locators")
        if start_contains == end_contains:
            raise ValueError("interval boundaries require independent source locators")
        segments = [locate(lambda text, needle=needle: needle.casefold() in text.casefold())
                    for needle in (start_contains, end_contains)]
        value = []
        for segment in segments:
            candidates = re.findall(r"\b[A-Za-z]+\s+\d{1,2},\s*\d{4}\b|\b\d{4}-\d{2}-\d{2}\b", segment.text)
            dates = {date_value(candidate) for candidate in candidates} - {None}
            if len(dates) != 1:
                raise ValueError("source boundary does not contain one unambiguous date")
            value.append(next(iter(dates)))
        if value[0] > value[1]:
            raise ValueError("source interval is reversed")
        value_type = "interval"
    elif mode == "labeled_date_interval":
        if not label or first_comma_part or start_contains or end_contains:
            raise ValueError("labeled interval needs one explicit source field label")
        pattern = re.compile(r"^\s*" + re.escape(label) + r"\s*[:：]\s*(.+)$", re.IGNORECASE)
        segment = locate(lambda text: bool(pattern.fullmatch(text)))
        text = pattern.fullmatch(segment.text).group(1)
        dates = list(re.finditer(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}\s+[A-Za-z]+\s+\d{4}\b|\b[A-Za-z]+\s+\d{1,2},\s*\d{4}\b", text))
        if len(dates) != 2 or not re.fullmatch(r"\s*(?:to|[-–—])\s*", text[dates[0].end():dates[1].start()], re.IGNORECASE):
            raise ValueError("source interval does not contain two unambiguous range boundaries")
        value = [date_value(match.group()) for match in dates]
        if not all(value) or value[0] > value[1]:
            raise ValueError("source interval is ambiguous or reversed")
        segments, value_type = [segment], "interval"
    elif mode == "labeled_identifier_set":
        if not label or first_comma_part or start_contains or end_contains:
            raise ValueError("identifier set needs one explicit source field label")
        pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(label)
                             + r"\s*[:：]\s*([A-Za-z0-9][A-Za-z0-9._/-]{2,63})(?![A-Za-z0-9])", re.IGNORECASE)
        matches = [(segment, pattern.findall(segment.text)) for segment in material.segments if pattern.search(segment.text)]
        if not matches:
            raise ValueError("source identifier field is absent")
        segments, value = [], []
        for segment, identifiers in matches:
            if not any(normalize(segment.text) == normalize(text) for text in paragraphs):
                raise ValueError("parsed identifier row is not present in the original DOCX")
            segments.append(segment)
            value.extend(identifiers)
        if len(value) != len(set(value)):
            raise ValueError("source identifier rows contain duplicate identifiers")
        value, value_type = sorted(value), "set"
    else:
        raise ValueError("unknown original text-field mode")
    observation = Operand.model_validate({"condition_id": condition_id, "role": role,
        "evidence_id": evidence_id, "value": value, "value_type": value_type,
        "status": "obtained", "depends_on": [evidence_id],
        "refs": [{"evidence_id": evidence_id, "segment_id": s.segment_id, "quote": s.text}
                 for s in dict((s.segment_id, s) for s in segments).values()]})
    case.model_copy(update={"operands": [observation]}).check_links()
    return observation, {"reader": "original_docx_named_fields_v1", "source_sha256": material.source_sha256,
        "evidence_id": evidence_id, "mode": mode, "label": label, "first_comma_part": first_comma_part,
        "start_contains": start_contains, "end_contains": end_contains,
        "source_segments": [s.segment_id for s in segments], "condition_id": condition_id, "role": role}
