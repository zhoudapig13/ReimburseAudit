import importlib.util
from pathlib import Path

import pytest

from audit_engine.examples.patent_fixture import sample_case


spec = importlib.util.spec_from_file_location(
    "evaluate_current_system_full",
    Path(__file__).resolve().parents[1] / "code/scripts/evaluate_current_system_full.py",
)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def record(compiled, read):
    return {"case": read.model_dump(), "case_sha256": read.digest(),
            "source_case_sha256": compiled.digest()}


def test_actual_readings_cannot_replace_new_policies_with_stale_conditions():
    original = sample_case()
    changed = original.model_copy(deep=True)
    changed.conditions[0].operation.right_constant = "80"
    with pytest.raises(ValueError, match="different compiled input"):
        evaluation.merge_reader_case(changed, record(original, original))
    # A forged source-case binding still cannot replace reviewed policies.
    with pytest.raises(ValueError, match="compiled policies"):
        evaluation.merge_reader_case(changed, record(changed, original))


def test_reader_may_update_read_status_but_cannot_edit_source_content():
    original = sample_case()
    read = original.model_copy(deep=True)
    read.input_version = "patent_reader_v1:actual_call_hash"
    read.evidence[0].read_status = "partial"
    assert evaluation.merge_reader_case(original, record(original, read)) == read
    read.evidence[0].text = "Changed source text"
    with pytest.raises(ValueError, match="source material"):
        evaluation.merge_reader_case(original, record(original, read))
