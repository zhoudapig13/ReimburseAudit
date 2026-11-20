import copy
import hashlib
from decimal import Decimal

import pytest
import torch

from audit_engine.contracts import AuditCase
from audit_engine.examples.patent_fixture import sample_case
from audit_engine.inference.runner import scalar_policies
from audit_engine.policy.amounts import project_amounts
from audit_engine.policy.programs import BoundMoneyExpression, compute_amount_units


def expression(bindings, node):
    return {"bindings": bindings, "expression": node}


def programmed_case():
    raw = sample_case().model_dump()
    raw["policy_numbers"] = [{"parameter_id": "night_cap", "value": "100", "currency": "CNY",
        "refs": raw["conditions"][0]["policy_refs"], "compile_status": "verified", "verification_method": "synthetic_source_fixture"}]
    limit = expression({"cap": {"kind": "policy_number", "parameter_id": "night_cap"}}, {"op": "VALUE", "source": "cap"})
    basis = expression({"invoice": {"kind": "operand", "condition_id": "cap", "role": "nightly"}}, {"op": "VALUE", "source": "invoice"})
    raw["conditions"][0]["roles"].append("calculated_cap")
    raw["conditions"][0]["derived_roles"] = {"calculated_cap": limit}
    raw["conditions"][0]["operation"].update(right_from_policy=False, right_constant=None, right_role="calculated_cap")
    raw["amount_programs"] = [{"object_id": "stay", "basis": basis, "limit": limit,
        "compile_status": "verified", "verification_method": "synthetic_source_fixture"}]
    raw["amount_units"][0].update(limit_status="unknown", limit=None, material_supported_max=None)
    return AuditCase.model_validate(raw)


def test_actual_amount_source_and_policy_limit_drive_projection():
    case = programmed_case()
    units, issues, traces = compute_amount_units(case, {"stay": "accept"})
    result = project_amounts(units, {"stay": "accept"}, "90.00", "CNY")
    assert not issues and result["A_ok"] == "90.00" and Decimal(result["A_max"]) == 90
    assert set(traces[0]["source_ids"]) == {"receipt", "policy"}


def test_masked_policy_parameter_cannot_survive_in_derived_comparison():
    case = programmed_case()
    policy = scalar_policies(case)[0]
    assert policy.rebuild(torch.tensor([True, True]), 8)[3].any()
    assert not policy.rebuild(torch.tensor([True, False]), 8)[3].any()
    assert not policy.rebuild(torch.tensor([False, True]), 8)[3].any()


def test_equivalent_amount_copy_survives_mask_without_restoring_deleted_source():
    raw = programmed_case().model_dump()
    duplicate = copy.deepcopy(raw["evidence"][0])
    duplicate.update(evidence_id="copy", raw_ref="copy.txt")
    raw["evidence"].append(duplicate)
    raw["application"]["attachment_ids"].append("copy")
    observed = copy.deepcopy(raw["operands"][0])
    observed.update(evidence_id="copy", depends_on=["copy"], value="90")
    observed["refs"][0]["evidence_id"] = "copy"
    raw["operands"].append(observed)
    case = AuditCase.model_validate(raw)
    value = BoundMoneyExpression(case, case.amount_programs[0].basis).evaluate([False, True, True])
    assert value.amount == 90 and value.sources == frozenset({"copy"})


def test_arbitrary_operation_and_injected_derived_value_are_rejected():
    raw = programmed_case().model_dump()
    raw["amount_programs"][0]["basis"]["expression"] = {"op": "EVAL", "args": []}
    with pytest.raises(ValueError, match="whitelist"):
        AuditCase.model_validate(raw)
    raw = programmed_case().model_dump()
    raw["operands"][0]["role"] = "calculated_cap"
    with pytest.raises(ValueError, match="must be recomputed"):
        AuditCase.model_validate(raw)


def test_unknown_currency_keeps_amount_for_review():
    raw = programmed_case().model_dump()
    raw["operands"][0]["currency"] = None
    units, issues, _ = compute_amount_units(AuditCase.model_validate(raw), {"stay": "accept"})
    assert issues and units[0].material_supported_max is None and units[0].limit_status == "unknown"


def test_policy_amount_can_end_with_sentence_punctuation():
    raw = programmed_case().model_dump()
    raw["evidence"][1]["segments"][0]["text"] = "Nightly cap 100.00."
    raw["policy_numbers"][0]["refs"][0]["quote"] = "cap 100.00."
    assert AuditCase.model_validate(raw).policy_numbers[0].value == "100"


def test_signature_agreement_is_a_real_source_for_roles_citations_and_masking():
    raw = programmed_case().model_dump()
    for identifier, sheet, rows in [
        ("days", "Days", ["A1=ID | B1=Days", "A2=P001 | B2=2"]),
        ("ack", "Signed", ["A1=ID | B1=Days | C1=Signature", "A2=P001 | B2=2 | C2=Person (signed)"]),
    ]:
        text = "\n".join(rows)
        raw["evidence"].append({"evidence_id": identifier, "type": "xlsx", "raw_ref": identifier + ".xlsx",
            "source_sha256": hashlib.sha256(text.encode()).hexdigest(), "read_status": "obtained",
            "segments": [{"segment_id": identifier + str(i), "text": "Sheet " + sheet + ": " + row}
                         for i, row in enumerate(rows)]})
        raw["application"]["attachment_ids"].append(identifier)
    limit = {"bindings": {
        "cap": {"kind": "policy_number", "parameter_id": "night_cap"},
        "days": {"kind": "table_sum", "evidence_id": "days", "sheet": "Days", "column": "Days",
                 "row_keys": ["ID"], "group_key": "ID"},
        "ack": {"kind": "table_sum", "evidence_id": "ack", "sheet": "Signed", "column": "Days",
                "row_keys": ["ID"], "group_key": "ID", "required_contains": {"Signature": "(signed)"}},
    }, "expression": {"op": "VALUE", "source": "cap", "quantity_source": "days"},
        "table_agreements": [["days", "ack"]]}
    raw["conditions"][0]["derived_roles"]["calculated_cap"] = limit
    raw["amount_programs"][0]["limit"] = limit
    case = AuditCase.model_validate(raw)
    value = BoundMoneyExpression(case, case.amount_programs[0].limit).evaluate()
    assert value.amount == 200 and value.sources == frozenset({"policy", "days", "ack"})
    assert "ack" in {ref.evidence_id for ref in value.refs}
    policy = scalar_policies(case)[0]
    role_map, _, _, _ = policy.rebuild(torch.ones(4, dtype=torch.bool), 8)
    assert role_map[1, 3]
    with pytest.raises(ValueError, match="masked"):
        BoundMoneyExpression(case, case.amount_programs[0].limit).evaluate([True, True, True, False])
    assert not policy.rebuild(torch.tensor([True, True, True, False]), 8)[3].any()
