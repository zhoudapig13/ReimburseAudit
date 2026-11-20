from datetime import date

import pytest

from audit_engine.contracts import NumberBinding
from audit_engine.examples.patent_fixture import sample_case
from audit_engine.policy.table_numbers import sum_source_table


def table(rows):
    original = sample_case().evidence[0].model_dump()
    original.update(type="xlsx", raw_ref="attendance.xlsx", text="\n".join(rows))
    original["segments"] = [
        {"segment_id": f"row_{i}", "text": row}
        for i, row in enumerate(rows)
    ]
    return type(sample_case().evidence[0]).model_validate(original)


def binding(**changes):
    return NumberBinding.model_validate({
        "kind": "table_sum", "evidence_id": "receipt", "sheet": "Attendance",
        "column": "Attendance", "row_keys": ["Date", "ID"],
        "date_key": "Date", "group_key": "ID", "allowed_values": ["0", "1"],
        **changes,
    })


HEADER = "Sheet Attendance: A1=Date | B1=ID | C1=Attendance"


def test_period_filter_keeps_source_identity_and_ignores_footer():
    source = table([
        HEADER,
        "Sheet Attendance: A2=2025-04-01 | B2=123456 | C2=1",
        "Sheet Attendance: A3=2025-04-02 | B3=123456 | C3=0",
        "Sheet Attendance: A4=2025-05-01 | B4=123456 | C4=1",
        "Sheet Attendance: A5=Total | C5=2",
    ])
    total, groups, refs = sum_source_table(
        source, binding(), (date(2025, 4, 1), date(2025, 4, 30))
    )
    assert total == 1 and groups == {"123456": 1}
    assert {r.segment_id for r in refs} == {"row_0", "row_1", "row_2"}


@pytest.mark.parametrize("row", [
    "Sheet Attendance: A3=2025-04-01 | B3=123456 | C3=1",
    "Sheet Attendance: A3=2025-04-02 | B3=123456 | C3=0.5",
    "Sheet Attendance: A3=2025-04-02 | B3=123456",
    "Sheet Attendance: A3=2025-04-02 | C3=1",
])
def test_duplicate_or_incomplete_or_invalid_attendance_cannot_be_summed(row):
    source = table([HEADER, "Sheet Attendance: A2=2025-04-01 | B2=123456 | C2=1", row])
    with pytest.raises(ValueError):
        sum_source_table(source, binding())


def test_empty_period_is_zero_but_header_without_data_is_unresolved():
    source = table([HEADER, "Sheet Attendance: A2=2025-05-01 | B2=123456 | C2=1"])
    assert sum_source_table(source, binding(), (date(2025, 4, 1), date(2025, 4, 30)))[:2] == (0, {})
    with pytest.raises(ValueError, match="no identified"):
        sum_source_table(table([HEADER]), binding())


def test_signature_is_required_when_declared():
    source = table([
        HEADER + " | D1=Signature",
        "Sheet Attendance: A2=2025-04-01 | B2=123456 | C2=1 | D2=unconfirmed",
    ])
    with pytest.raises(ValueError, match="acknowledgement"):
        sum_source_table(source, binding(required_contains={"Signature": "(signed)"}))
