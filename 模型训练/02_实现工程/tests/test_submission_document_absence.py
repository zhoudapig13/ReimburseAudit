import copy

import pytest
import torch

from audit_engine.contracts import AuditCase
from audit_engine.examples.patent_fixture import sample_case
from audit_engine.inference.runner import AuditRunner, CaseFeatures
from audit_engine.models.heads import AuditHeads


def missing_document_case():
    raw = sample_case().model_dump()
    raw["operands"] = []
    raw["conditions"][0]["consequence"]["on_gap"] = "reject_object"
    raw["document_absences"] = [{"condition_id": "cap", "role": "nightly",
        "method": "full_submission_document_absence_v1", "reader_signature_sha256": "a" * 64,
        "sources": [{"evidence_id": e["evidence_id"], "source_sha256": e["source_sha256"],
                     "checked_segment_ids": [s["segment_id"] for s in e["segments"]],
                     "notes": ["Synthetic complete source: no requested transactional document."],
                     "raw_call_sha256": ["b" * 64]} for e in raw["evidence"]]}]
    return raw


@pytest.mark.parametrize("mutation", ["missing_source", "changed_source", "partial_text", "competing_read", "new_attachment"])
def test_absence_cannot_survive_incomplete_coverage_or_conflicting_evidence(mutation):
    raw = missing_document_case()
    if mutation == "missing_source":
        raw["document_absences"][0]["sources"].pop()
    elif mutation == "changed_source":
        raw["evidence"][0]["source_sha256"] = "c" * 64
    elif mutation == "partial_text":
        raw["document_absences"][0]["sources"][0]["checked_segment_ids"] = []
    elif mutation == "competing_read":
        raw["operands"] = sample_case().model_dump()["operands"]
    else:
        source = copy.deepcopy(raw["evidence"][0])
        source["evidence_id"] = "new_proof"
        raw["evidence"].append(source)
        raw["application"]["attachment_ids"].append("new_proof")
    with pytest.raises(ValueError, match="document absence"):
        AuditCase.model_validate(raw)


@pytest.mark.parametrize("material_type", ["image", "pdf"])
def test_whole_image_and_all_pdf_pages_are_required(material_type):
    raw = missing_document_case()
    raw["evidence"][0]["type"] = material_type
    check = raw["document_absences"][0]["sources"][0]
    if material_type == "pdf":
        check.update(page_count=2, checked_pages=[1])
    with pytest.raises(ValueError, match="whole image|all PDF pages"):
        AuditCase.model_validate(raw)


def test_confirmed_missing_document_uses_gap_branch_without_internal_fact_repair():
    torch.manual_seed(7)
    model = AuditHeads(16, 8, 6, 2)
    raw = missing_document_case()
    for include_proof in [False, True]:
        candidate = copy.deepcopy(raw)
        if not include_proof:
            candidate["document_absences"] = []
        case = AuditCase.model_validate(candidate)
        features = CaseFeatures(case.digest(), [e.evidence_id for e in case.evidence], ["cap"],
            torch.randn(2, 16), torch.randn(16), torch.randn(1, 16), {"pooling": "last_non_padding_input_v1"})
        result = AuditRunner(model).run(case, features)
        assert result["F"][0]["state"] == "insufficient"
        assert result["F"][0]["G"]
        assert result["module_calls"]["relation"] > 0
        assert result["module_calls"]["contribution"] > 0
        assert result["module_calls"]["corrector"] == 0
        assert result["answer"]["max_legal"] == ("0" if include_proof else None)
