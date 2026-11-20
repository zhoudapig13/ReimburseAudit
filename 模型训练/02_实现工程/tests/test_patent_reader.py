import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from audit_engine.contracts import AuditCase
from audit_engine.examples.patent_fixture import sample_case
from audit_engine.readers.patent_reader import PatentReader, source_views


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, tokens):
        return "".join(map(chr, tokens))


class FakeEncoder:
    signature = {"model": "fixture", "revision": "fixture"}
    processor = SimpleNamespace(tokenizer=Tokenizer())

    def __init__(self, response):
        self.response = response
        self.calls = 0

    def read(self, prompt, image, max_new_tokens):
        self.calls += 1
        assert "reference_hint" not in prompt
        return json.dumps(self.response)


def setup_reader(tmp_path):
    case = sample_case()
    (tmp_path / "receipt.txt").write_text("Lodging 90.00")
    (tmp_path / "policy.txt").write_text("Nightly cap 100")
    condition = case.conditions[0]
    proposal = {"question": condition.question, "roles": condition.roles,
                "operation": condition.operation.model_dump(), "source_segment_ids": ["p1"],
                "on_failed": "cap_excess"}
    encoder = FakeEncoder({"rules": [proposal], "issues": []})
    return case, PatentReader(encoder, tmp_path, tmp_path / "calls")


def test_pending_policy_is_cited_and_source_cache_is_reused(tmp_path):
    case, reader = setup_reader(tmp_path)
    source = case.evidence[1]
    view = next(source_views(case, source, Tokenizer(), 200))
    result = reader.propose(case, source, view)
    assert result.conditions[0].compile_status == "pending"
    other = case.model_copy(update={"workflow_id": "another"})
    reader.propose(other, source, view)
    assert reader.encoder.calls == 1
    assert list((tmp_path / "calls").glob("*.json"))


@pytest.mark.parametrize("value,note,rejected", [
    ("complete", "The consumption date is not printed, but other fields are present.", True),
    ("complete", "No date appears on the original receipt.", True),
    ("complete", "Although the consumption date is not explicitly printed, the receipt is considered complete.", True),
    ("incomplete", "The consumption date is not printed.", False),
    ("complete", "Merchant name, address, consumption date, itemized expenses and total are visible.", False),
])
def test_completeness_claim_must_agree_with_its_own_observation(tmp_path, value, note, rejected):
    case, reader = setup_reader(tmp_path)
    raw = case.model_dump()
    raw["conditions"][0].update(roles=["required_fields_status"],
        role_definitions={"required_fields_status": "All fields must be present: merchant name, address, consumption date, itemized expenses and total."},
        operation={"name": "EQ", "left_role": "required_fields_status", "right_from_policy": True,
                   "right_constant": "complete", "value_type": "text"})
    raw["operands"] = []
    raw["evidence"][0]["type"] = "image"
    case = AuditCase.model_validate(raw)
    reader.encoder.response = {"facts": [{"role": "required_fields_status", "value": value,
        "value_type": "text", "status": "obtained", "note": note}], "issues": []}
    view = {"kind": "image", "source_sha256": case.evidence[0].source_sha256}
    if rejected:
        with pytest.raises(ValueError, match="completeness contradicts"):
            reader.read_roles(case, case.evidence[0], case.conditions[0], view)
    else:
        result = reader.read_roles(case, case.evidence[0], case.conditions[0], view)
        assert result.operands[0].value == value


def test_reader_cannot_promote_policy_or_cite_another_source(tmp_path):
    case, reader = setup_reader(tmp_path)
    source = case.evidence[1]
    view = next(source_views(case, source, Tokenizer(), 200))
    reader.encoder.response["rules"][0]["compile_status"] = "verified"
    result = reader.propose(case, source, view)
    assert not result.conditions and any("Extra inputs" in issue for issue in result.issues)


def test_policy_atom_cannot_cite_a_segment_outside_its_source_view(tmp_path):
    case, reader = setup_reader(tmp_path)
    source = case.evidence[1]
    view = next(source_views(case, source, Tokenizer(), 200))
    reader.encoder.response["rules"][0]["source_segment_ids"] = ["r1"]
    with pytest.raises(ValueError, match="outside"):
        reader.propose(case, source, view)


def test_policy_atom_binds_constant_kind_without_changing_numeric_value():
    from audit_engine.readers.patent_reader import PolicyAtom

    atom = PolicyAtom.model_validate({"question": "Is the fee within cap?", "roles": ["actual_fee"],
        "source_segment_ids": ["p1"], "operation": {"name": "LE", "left_role": "actual_fee",
        "right_constant": "¥1,200.00", "right_from_policy": False, "value_type": "money", "currency": "CNY"}})
    assert atom.operation.right_constant == "1200.00" and atom.operation.right_from_policy


def test_policy_atom_cannot_replace_a_real_check_with_self_comparison():
    from audit_engine.readers.patent_reader import PolicyAtom
    with pytest.raises(ValueError, match="itself"):
        PolicyAtom.model_validate({"question": "Does the daily rate match the approved rate?",
            "roles": ["actual_role"], "source_segment_ids": ["p1"],
            "operation": {"name": "EQ", "left_role": "actual_role", "right_role": "actual_role", "value_type": "text"}})


def test_numeric_policy_constant_must_be_in_its_own_cited_source(tmp_path):
    case, reader = setup_reader(tmp_path)
    reader.encoder.response["rules"][0]["operation"]["right_constant"] = "500"
    material = case.evidence[1]
    view = next(source_views(case, material, Tokenizer(), 200))
    result = reader.propose(case, material, view)
    assert not result.conditions
    assert any("unbound_numeric_constant" in issue for issue in result.issues)


def test_policy_proposals_do_not_keep_dangling_item_allocations_or_approved_coverage(tmp_path):
    path = Path(__file__).resolve().parents[1] / "code/scripts/read_patent_cases.py"
    spec = importlib.util.spec_from_file_location("proposal_scope", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    case, reader = setup_reader(tmp_path)
    original = case.model_dump()
    result = module.process_case(reader, case, "propose", {"reader": {"text_chunk_tokens": 200}}, {})
    pending = AuditCase.model_validate(result["case"])
    assert pending.policy_coverage == "pending"
    assert pending.amount_units[0].object_id == "claim"
    assert not pending.amount_programs and not pending.policy_numbers
    assert pending.conditions and all(c.object_id == "claim" and c.compile_status == "pending" for c in pending.conditions)
    assert result["proposal_amount_scope"] == "aggregate_placeholder_pending_source_allocation"
    assert case.model_dump() == original


def test_views_pack_segments_and_preserve_their_source_text():
    case = sample_case()
    material = case.evidence[1]
    material = material.model_copy(
        update={
            "segments": [
                s.model_copy(update={"segment_id": f"s{i}", "text": "abc"})
                for i, s in enumerate([material.segments[0]] * 4)
            ]
        }
    )
    views = list(source_views(case, material, Tokenizer(), 60))
    segments = [s for v in views for s in v["segments"]]
    assert [s["segment_id"] for s in segments] == ["s0", "s1", "s2", "s3"]
    assert len(views) == 2


def test_raw_source_change_invalidates_reader(tmp_path):
    case, reader = setup_reader(tmp_path)
    (tmp_path / "policy.txt").write_text("changed")
    with pytest.raises(ValueError, match="raw source changed"):
        reader.source(case.evidence[1])


def test_text_observation_note_cannot_replace_a_literal_source_quote(tmp_path):
    case, reader = setup_reader(tmp_path)
    reader.encoder.response = {"facts": [{"role": "nightly", "value": "90.00",
        "value_type": "money", "currency": "CNY", "note": "The source says 90.00",
        "segment_id": "r1", "quote": ""}], "issues": []}
    view = next(source_views(case, case.evidence[0], Tokenizer(), 200))
    with pytest.raises(ValueError, match="verbatim quote"):
        reader.read_roles(case, case.evidence[0], case.conditions[0], view)


def test_explicit_category_role_rejects_a_paragraph_in_place_of_its_label(tmp_path):
    from audit_engine.readers.patent_reader import declared_categories
    case, reader = setup_reader(tmp_path)
    raw = case.model_dump()
    raw["conditions"][0].update(roles=["category"], operation=None,
        role_definitions={"category": "Classify the stated cause as travel_expense, personal_expense, or no_explanation. Read the original source."})
    raw["operands"] = []
    case = AuditCase.model_validate(raw)
    assert declared_categories(case.conditions[0].role_definitions["category"]) == (
        "travel_expense", "personal_expense", "no_explanation")
    assert not declared_categories("Read the original date as YYYY-MM-DD.")
    reader.encoder.response = {"facts": [{"role": "category", "value": "The source describes lodging.",
        "value_type": "text", "quote": "Lodging 90.00", "segment_id": "r1"}], "issues": []}
    view = next(source_views(case, case.evidence[0], Tokenizer(), 200))
    with pytest.raises(ValueError, match="declares the categories"):
        reader.read_roles(case, case.evidence[0], case.conditions[0], view)


def test_runtime_treats_legacy_invalid_category_as_a_gap_instead_of_a_violation():
    import torch

    from audit_engine.inference.runner import scalar_policies
    raw = sample_case().model_dump()
    raw["conditions"][0].update(roles=["nightly"],
        role_definitions={"nightly": "Classify the expense as travel_expense, personal_expense, or no_explanation."},
        operation={"name": "EQ", "left_role": "nightly", "right_from_policy": True,
                   "right_constant": "travel_expense", "value_type": "text"})
    raw["operands"][0].update(value="The source describes lodging.", value_type="text", currency=None)
    case = AuditCase.model_validate(raw)
    policy = scalar_policies(case)[0]
    roles, _, comparison, complete = policy.rebuild(torch.ones(len(case.evidence), dtype=torch.bool), 8)
    assert not roles.any() and not complete.any()
    assert comparison.argmax(-1).tolist() == [2, 2]
    raw["operands"][0]["value"] = "travel_expense"
    valid = scalar_policies(AuditCase.model_validate(raw))[0]
    assert valid.rebuild(torch.ones(len(case.evidence), dtype=torch.bool), 8)[3].any()


def test_role_reader_checks_role_and_visual_citation_type(tmp_path):
    case, reader = setup_reader(tmp_path)
    reader.encoder.response = {
        "facts": [
            {
                **{
                    k: v
                    for k, v in case.operands[0].model_dump().items()
                    if k in {"role", "value", "value_type", "status", "currency"}
                },
                **{
                    k: v
                    for k, v in case.operands[0].refs[0].model_dump().items()
                    if k in {"segment_id", "quote", "note"}
                },
            }
        ],
        "issues": [],
    }
    view = next(source_views(case, case.evidence[0], Tokenizer(), 200))
    output = reader.read_roles(case, case.evidence[0], case.conditions[0], view)
    assert output.operands[0].value == "90.00"
    reader.encoder.response = copy.deepcopy(reader.encoder.response)
    reader.encoder.response["facts"][0]["role"] = "invented_role"
    reader.output_root = tmp_path / "different_cache"
    reader.output_root.mkdir()
    with pytest.raises(ValueError, match="undeclared roles"):
        reader.read_roles(case, case.evidence[0], case.conditions[0], view)


def test_pilot_selection_is_group_diverse_and_split_bound():
    path = Path(__file__).resolve().parents[1] / "code/scripts/read_patent_cases.py"
    spec = importlib.util.spec_from_file_location("read_patent_cases", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = [sample_case(str(i)) for i in range(5)]
    partitions = {
        str(i): {
            "split": "train" if i < 4 else "test",
            "source_family_id": "g" + str(i // 2),
        }
        for i in range(5)
    }
    selected = module.select_pilot(cases, partitions, "train", 2)
    assert len({partitions[c.workflow_id]["source_family_id"] for c in selected}) == 2
    assert all(partitions[c.workflow_id]["split"] == "train" for c in selected)


def test_timestamp_policy_preserves_minute_level_violation_and_timezone_gaps():
    from audit_engine.policy.comparators import evaluate

    late = evaluate(
        "LE", "2024-10-17T19:01:00", "2024-10-17T19:00:00", value_type="datetime"
    )
    assert late.complete and not late.passed
    valid = evaluate(
        "WITHIN",
        "2024-10-17T08:32:00",
        ("2024-10-17T08:00:00", "2024-10-17T19:00:00"),
        value_type="datetime",
    )
    assert valid.complete and valid.passed
    unknown = evaluate(
        "LE", "2024-10-17T19:01:00+00:00", "2024-10-17T19:00:00", value_type="datetime"
    )
    assert not unknown.complete


def test_equal_decimal_observations_are_not_a_source_contradiction():
    from audit_engine.inference.runner import scalar_policies

    raw = sample_case().model_dump()
    duplicate = copy.deepcopy(raw["evidence"][0])
    duplicate["evidence_id"] = "receipt_copy"
    duplicate["raw_ref"] = "receipt_copy.txt"
    raw["evidence"].append(duplicate)
    raw["application"]["attachment_ids"].append("receipt_copy")
    observed = copy.deepcopy(raw["operands"][0])
    observed.update(evidence_id="receipt_copy", depends_on=["receipt_copy"], value="90")
    observed["refs"][0]["evidence_id"] = "receipt_copy"
    raw["operands"].append(observed)
    case = AuditCase.model_validate(raw)
    policy = scalar_policies(case)[0]
    assert policy.values[0][0] == policy.values[0][2] == "90"


def test_approved_role_origin_excludes_policy_from_actual_receipt_read(tmp_path):
    case, reader = setup_reader(tmp_path)
    condition = case.conditions[0].model_copy(
        update={"role_origins": {"nightly": "receipt"}}
    )
    case = case.model_copy(update={"conditions": [condition]})
    material = case.evidence[1]
    view = next(source_views(case, material, Tokenizer(), 200))
    assert reader.read_roles(case, material, condition, view).operands == []
    assert reader.encoder.calls == 0


def test_source_date_normalization_preserves_numeric_date_uncertainty():
    from audit_engine.readers.value_formats import normalize_fact_value

    value, status = normalize_fact_value(
        "October 10, 2012 – October 12, 2012", "interval", "obtained"
    )
    assert value == ["2012-10-10", "2012-10-12"] and status == "obtained"
    value, status = normalize_fact_value("09/04/2021", "date", "obtained")
    assert value == "09/04/2021" and status == "ambiguous"


def test_visual_transcription_is_bound_to_whole_image_not_fake_text_segment(tmp_path):
    case, reader = setup_reader(tmp_path)
    material = case.evidence[0].model_copy(update={"type": "image"})
    case = case.model_copy(update={"evidence": [material, case.evidence[1]]})
    reader.encoder.response = {
        "facts": [
            {
                "role": "nightly",
                "value": "90.00",
                "value_type": "money",
                "currency": "CNY",
                "quote": "90.00",
            }
        ]
    }
    output = reader.read_roles(
        case,
        material,
        case.conditions[0],
        {"kind": "image"},
        image=tmp_path / "receipt.txt",
    )
    ref = output.operands[0].refs[0]
    assert ref.kind == "visual" and ref.segment_id is None and ref.quote == ""
    assert "90.00" in ref.note and output.operands[0].depends_on == ["receipt"]
