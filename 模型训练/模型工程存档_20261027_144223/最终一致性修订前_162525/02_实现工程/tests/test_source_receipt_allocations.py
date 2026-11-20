import copy

import pytest

from audit_engine.contracts import AuditCase
from audit_engine.examples.patent_fixture import sample_case
from audit_engine.policy.allocations import allocate_receipt_units
from audit_engine.policy.amounts import project_amounts


def two_receipts():
    raw = sample_case().model_dump()
    raw["input_version"] = "patent_reader_v1:fixture"
    raw["application"]["claimed_amount"] = "180"
    raw["amount_units"][0]["claimed_amount"] = "180"
    receipt = copy.deepcopy(raw["evidence"][0])
    receipt.update(evidence_id="receipt2", raw_ref="receipt2.txt")
    raw["evidence"].append(receipt)
    raw["application"]["attachment_ids"].append("receipt2")
    raw["application"]["receipt_owners"] = [{"owner": "claimant", "evidence_ids": ["receipt", "receipt2"]}]
    condition = copy.deepcopy(raw["conditions"][0])
    condition["condition_id"] = "cap2"
    raw["conditions"].append(condition)
    operand = copy.deepcopy(raw["operands"][0])
    operand.update(condition_id="cap2", evidence_id="receipt2", depends_on=["receipt2"])
    operand["refs"][0]["evidence_id"] = "receipt2"
    raw["operands"].append(operand)
    return AuditCase.model_validate(raw), [
        {"object_id": "outbound", "evidence_id": "receipt", "amount_condition_id": "cap",
         "amount_role": "nightly", "condition_ids": ["cap"]},
        {"object_id": "return", "evidence_id": "receipt2", "amount_condition_id": "cap2",
         "amount_role": "nightly", "condition_ids": ["cap2"]},
    ]


def test_ticket_rejection_does_not_discard_another_ticket():
    original, plan = two_receipts()
    allocated, provenance = allocate_receipt_units(original, plan)
    assert allocated.conditions[0].object_id == "outbound"
    assert allocated.conditions[1].object_id == "return"
    assert provenance["source_receipts_exactly_partition_claim"]
    assert allocated.evidence == original.evidence and allocated.operands == original.operands
    # Allocation does not manufacture legal caps or approval for the other unit.
    amounts = project_amounts(allocated.amount_units, {"outbound": "reject", "return": "review"}, "180", "CNY")
    assert amounts["A_no"] == "90.00" and amounts["A_review"] == "90.00"
    assert amounts["A_max"] is None


def test_unaccounted_claim_and_unconfirmed_currency_cannot_be_split():
    original, plan = two_receipts()
    raw = original.model_dump()
    raw["application"]["claimed_amount"] = raw["amount_units"][0]["claimed_amount"] = "181"
    with pytest.raises(ValueError, match="exactly account"):
        allocate_receipt_units(AuditCase.model_validate(raw), plan)
    raw = original.model_dump()
    raw["operands"][1]["currency"] = None
    with pytest.raises(ValueError, match="amount source"):
        allocate_receipt_units(AuditCase.model_validate(raw), plan)


def test_allocation_cannot_reuse_one_receipt_or_drop_conditions():
    original, plan = two_receipts()
    plan[1]["evidence_id"] = "receipt"
    with pytest.raises(ValueError, match="distinct"):
        allocate_receipt_units(original, plan)
    _, plan = two_receipts()
    with pytest.raises(ValueError, match="unassigned"):
        allocate_receipt_units(original, plan[:1])
