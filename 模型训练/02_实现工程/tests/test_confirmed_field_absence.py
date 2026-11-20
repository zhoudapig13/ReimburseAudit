import copy
import hashlib

import pytest
import torch
from PIL import Image

from audit_engine.contracts import AuditCase
from audit_engine.examples.patent_fixture import sample_case
from audit_engine.inference.runner import AuditRunner, CaseFeatures
from audit_engine.models.heads import AuditHeads
from audit_engine.readers.absence import bind_explicit_date_absence
from audit_engine.readers.value_formats import explicit_printed_date_absence


@pytest.mark.parametrize("note,confirmed", [
    ("No date is printed on the receipt.", True),
    ("The consumption date is not printed; other text is legible.", True),
    ("Although the consumption date is not explicitly printed, other fields are legible.", True),
    ("The date is not visible.", False),
    ("The image is blurred; no date is printed in the visible cropped area.", False),
    ("The printed date is 2025-01-01.", False),
])
def test_explicit_absence_differs_from_unreadable_source(note, confirmed):
    assert explicit_printed_date_absence(note) is confirmed


@pytest.mark.parametrize("height,note,bound", [
    (150, "The consumption date is not explicitly printed on the receipt.", True),
    (400, "No date is printed on the receipt.", False),
    (150, "The date is unreadable in the cropped image.", False),
])
def test_absence_binder_requires_whole_image_and_explicit_observation(tmp_path, height, note, bound):
    source = tmp_path / "receipt.png"
    Image.new("RGB", (100, height), "white").save(source)
    raw = sample_case().model_dump()
    raw["input_version"] = "patent_reader_v1:fixture"
    raw["evidence"][0].update(evidence_id="e006", type="image", raw_ref=source.name,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    raw["application"]["attachment_ids"][0] = "e006"
    condition = raw["conditions"][0]
    condition.update(roles=["actual_e006_printed_date"],
        role_definitions={"actual_e006_printed_date": "The calendar date actually printed on the original receipt."},
        operation={"name": "EQ", "left_role": "actual_e006_printed_date", "right_from_policy": True,
                   "right_constant": "2025-01-01", "value_type": "date"})
    auxiliary = copy.deepcopy(condition)
    auxiliary.update(condition_id="fields", roles=["required_fields"], operation=None,
                     role_definitions={"required_fields": "Merchant name, address and consumption date."})
    raw["conditions"].append(auxiliary)
    raw["operands"] = [{"condition_id": "fields", "role": "required_fields", "evidence_id": "e006",
        "value": "complete", "value_type": "text", "status": "obtained", "depends_on": ["e006"],
        "refs": [{"evidence_id": "e006", "kind": "visual", "note": note}]}]
    case = AuditCase.model_validate(raw)
    result, records = bind_explicit_date_absence(case, tmp_path)
    marker = "confirmed_missing:cap:actual_e006_printed_date"
    assert (marker in result.unresolved) is bound
    assert bool(records) is bound
    if bound:
        assert result.operands[-1].value is None
        assert result.operands[-1].refs[0].note == note
        assert result.operands[-1].status == "not_obtained"
        source.write_bytes(b"changed")
        with pytest.raises(ValueError, match="source changed"):
            bind_explicit_date_absence(case, tmp_path)


@pytest.mark.parametrize("marker,note,rejected", [
    (True, "No date is printed on this whole receipt.", True),
    (False, "No date is printed on this whole receipt.", False),
    (True, "The image is blurred and the date cannot be read.", False),
])
def test_policy_rejects_only_confirmed_missing_required_date(marker, note, rejected):
    raw = sample_case().model_dump()
    raw["evidence"][0]["type"] = "image"
    raw["conditions"][0].update(roles=["printed_date"], role_definitions={"printed_date": "Date actually printed on the original receipt."},
        operation={"name": "EQ", "left_role": "printed_date", "right_from_policy": True,
                   "right_constant": "2025-01-01", "value_type": "date"},
        consequence={"on_failed": "reject_object", "on_gap": "reject_object"})
    raw["operands"] = [{"condition_id": "cap", "role": "printed_date", "evidence_id": "receipt",
        "value": None, "value_type": "date", "status": "not_obtained", "depends_on": ["receipt"],
        "refs": [{"evidence_id": "receipt", "kind": "visual", "note": note}]}]
    raw["unresolved"] = ["confirmed_missing:cap:printed_date"] if marker else []
    case = AuditCase.model_validate(raw)
    torch.manual_seed(7)
    model = AuditHeads(16, 8, 6, 2)
    features = CaseFeatures(case.digest(), [e.evidence_id for e in case.evidence], ["cap"],
        torch.randn(2, 16), torch.randn(16), torch.randn(1, 16), {"pooling": "last_non_padding_input_v1"})
    result = AuditRunner(model).run(case, features)
    assert result["F"][0]["state"] == "insufficient"
    assert result["answer"]["max_legal"] == ("0" if rejected else None)
    assert (result["F"][0]["gaps"][0]["status"] == "confirmed_missing") is rejected
