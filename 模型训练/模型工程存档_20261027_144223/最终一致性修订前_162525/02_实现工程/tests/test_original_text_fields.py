import hashlib

import pytest
from docx import Document

from audit_engine.contracts import AuditCase
from audit_engine.examples.patent_fixture import sample_case
from audit_engine.readers.text_fields import read_original_text_field


def source_case(tmp_path, paragraphs):
    document = Document()
    for text in paragraphs:
        document.add_paragraph(text)
    source = tmp_path / "source.docx"
    document.save(source)
    raw = sample_case().model_dump()
    raw["evidence"][0].update(type="docx", raw_ref="source.docx",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        segments=[{"segment_id": "s" + str(i), "text": text} for i, text in enumerate(paragraphs)])
    raw["conditions"][0].update(roles=["city", "period"], operation=None, kind="open_text")
    raw["operands"] = []
    return AuditCase.model_validate(raw)


def test_fields_follow_original_source_instead_of_claim_or_expected_city(tmp_path):
    case = source_case(tmp_path, ["Meeting Location: Shanghai, China", "Original departure: May 23, 2024", "Actual departure: May 25, 2024"])
    city, provenance = read_original_text_field(case, tmp_path, evidence_id="receipt", condition_id="cap",
        role="city", mode="labeled_text", label="Meeting Location", first_comma_part=True)
    assert city.value == "Shanghai" and city.refs[0].quote == "Meeting Location: Shanghai, China"
    interval, _ = read_original_text_field(case, tmp_path, evidence_id="receipt", condition_id="cap",
        role="period", mode="date_interval", start_contains="Original departure:", end_contains="Actual departure:")
    assert interval.value == ["2024-05-23", "2024-05-25"]
    assert interval.depends_on == ["receipt"] and len(interval.refs) == 2
    assert provenance["source_sha256"] == case.evidence[0].source_sha256


def test_multiple_matching_fields_and_ambiguous_boundary_dates_are_not_chosen(tmp_path):
    case = source_case(tmp_path, ["Meeting Location: Shanghai", "Meeting Location: Beijing",
        "Original departure: May 23, 2024 or May 24, 2024", "Actual departure: May 25, 2024"])
    with pytest.raises(ValueError, match="ambiguous"):
        read_original_text_field(case, tmp_path, evidence_id="receipt", condition_id="cap", role="city",
            mode="labeled_text", label="Meeting Location")
    with pytest.raises(ValueError, match="unambiguous date"):
        read_original_text_field(case, tmp_path, evidence_id="receipt", condition_id="cap", role="period",
            mode="date_interval", start_contains="Original departure:", end_contains="Actual departure:")


def test_altered_original_document_is_rejected(tmp_path):
    case = source_case(tmp_path, ["Meeting Location: Shanghai"])
    (tmp_path / "source.docx").write_bytes(b"altered")
    with pytest.raises(ValueError, match="hash changed"):
        read_original_text_field(case, tmp_path, evidence_id="receipt", condition_id="cap", role="city",
            mode="labeled_text", label="Meeting Location")


def test_original_contract_interval_and_complete_identifier_set(tmp_path):
    case = source_case(tmp_path, ["Duration: 1 April 2025 – 30 April 2025 (30 days)",
        "No.: 1 | Name: Person A | ID Number: 001234567 | Role: Worker",
        "No.: 2 | Name: Person B | ID Number: 009876543 | Role: Worker"])
    period, _ = read_original_text_field(case, tmp_path, evidence_id="receipt", condition_id="cap",
        role="period", mode="labeled_date_interval", label="Duration")
    members, provenance = read_original_text_field(case, tmp_path, evidence_id="receipt", condition_id="cap",
        role="city", mode="labeled_identifier_set", label="ID Number")
    assert period.value == ["2025-04-01", "2025-04-30"]
    assert members.value == ["001234567", "009876543"]
    assert len(members.refs) == 2 and len(provenance["source_segments"]) == 2


def test_ambiguous_contract_range_and_duplicate_identifiers_are_not_silently_selected(tmp_path):
    case = source_case(tmp_path, ["Duration: 1 April 2025 or 30 April 2025",
        "Name: Person A | ID Number: 001234567", "Name: Person B | ID Number: 001234567"])
    with pytest.raises(ValueError, match="range boundaries"):
        read_original_text_field(case, tmp_path, evidence_id="receipt", condition_id="cap",
            role="period", mode="labeled_date_interval", label="Duration")
    with pytest.raises(ValueError, match="duplicate identifiers"):
        read_original_text_field(case, tmp_path, evidence_id="receipt", condition_id="cap",
            role="city", mode="labeled_identifier_set", label="ID Number")


def test_identifier_rows_are_verified_against_real_docx_table_cells(tmp_path):
    case = source_case(tmp_path, ["Placeholder"])
    document = Document(tmp_path / "source.docx")
    table = document.add_table(rows=3, cols=2)
    for row, values in zip(table.rows, [["Name", "ID Number"], ["Person A", "0012345"], ["Person B", "0098765"]]):
        for cell, value in zip(row.cells, values):
            cell.text = value
    document.save(tmp_path / "source.docx")
    raw = case.model_dump()
    raw["evidence"][0]["source_sha256"] = hashlib.sha256((tmp_path / "source.docx").read_bytes()).hexdigest()
    raw["evidence"][0]["segments"] = [
        {"segment_id": "a", "text": "Name: Person A | ID Number: 0012345"},
        {"segment_id": "b", "text": "Name: Person B | ID Number: 0098765"}]
    case = AuditCase.model_validate(raw)
    observation, _ = read_original_text_field(case, tmp_path, evidence_id="receipt", condition_id="cap",
        role="city", mode="labeled_identifier_set", label="ID Number")
    assert observation.value == ["0012345", "0098765"]
    raw["evidence"][0]["segments"][0]["text"] = "Name: Person A | ID Number: 7777777"
    with pytest.raises(ValueError, match="not present in the original DOCX"):
        read_original_text_field(AuditCase.model_validate(raw), tmp_path, evidence_id="receipt", condition_id="cap",
            role="city", mode="labeled_identifier_set", label="ID Number")
