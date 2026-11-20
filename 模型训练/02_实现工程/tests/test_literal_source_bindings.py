import pytest

from audit_engine.readers.literal_bindings import literal_source_refs
from audit_engine.readers.patent_reader import SourceFact


def test_serialized_collection_changes_structure_without_changing_members():
    fact = SourceFact.model_validate({"role": "prior_tickets", "value": '["G123456", "G123457"]',
        "value_type": "set", "quote": "G123456", "segment_id": "row2"})
    assert fact.value == ["G123456", "G123457"]
    interval = SourceFact.model_validate({"role": "delay_dates", "value": "['2013-01-19', '2013-01-20']",
        "value_type": "interval", "quote": "Original dates", "segment_id": "dates"})
    assert interval.value == ["2013-01-19", "2013-01-20"]


def test_cross_paragraph_observation_is_bound_to_separate_original_quotes():
    segments = [{"segment_id": "s1", "text": "The flight was delayed until the next morning."},
                {"segment_id": "s2", "text": "The flight finally departed January 20, 2013 at 02:00."}]
    refs = literal_source_refs("e1", segments,
        "The flight was delayed until the next morning. The flight finally departed January 20, 2013 at 02:00.",
        preferred_segment="wrong_id")
    assert [r["segment_id"] for r in refs] == ["s1", "s2"]
    assert [r["quote"] for r in refs] == [s["text"] for s in segments]
    with pytest.raises(ValueError, match="literal"):
        literal_source_refs("e1", segments, "The flight was delayed ... January 20, 2013 at 02:00.")


def test_binding_rejects_ambiguous_or_nonadjacent_source_fragments():
    duplicate = [{"segment_id": "a", "text": "The same original phrase."},
                 {"segment_id": "b", "text": "The same original phrase."}]
    with pytest.raises(ValueError, match="multiple segments"):
        literal_source_refs("e1", duplicate, "The same original phrase.")
    assert literal_source_refs("e1", duplicate, "The same original phrase.", "b")[0]["segment_id"] == "b"
    unrelated = [{"segment_id": "a", "text": "The approval was granted."},
                 {"segment_id": "middle", "text": "A different administrative event."},
                 {"segment_id": "c", "text": "The permit later expired."}]
    with pytest.raises(ValueError, match="literal"):
        literal_source_refs("e1", unrelated, "The approval was granted. The permit later expired.")
