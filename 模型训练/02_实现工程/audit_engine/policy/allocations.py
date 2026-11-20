"""Partition a claim only when actual receipt reads account for it exactly."""

import hashlib
import json
from decimal import Decimal

from audit_engine.contracts import AuditCase, MoneyExpression
from audit_engine.policy.programs import BoundMoneyExpression


def allocate_receipt_units(case: AuditCase, allocations: list[dict]):
    if len(case.amount_units) != 1 or case.amount_programs:
        raise ValueError("receipt allocation requires an unsplit claim without amount programs")
    if not case.input_version.startswith("patent_reader_v1:"):
        raise ValueError("receipt allocation requires actual source-reader inputs")
    owners = {key for row in case.application.receipt_owners for key in row.get("evidence_ids", [])}
    conditions = {condition.condition_id: condition for condition in case.conditions}
    assigned, objects, receipts, units, records = {}, set(), set(), [], []
    for allocation in allocations:
        object_id, evidence_id = allocation["object_id"], allocation["evidence_id"]
        if object_id in objects or evidence_id in receipts or evidence_id not in owners:
            raise ValueError("receipt allocations must name distinct declared receipt sources")
        objects.add(object_id)
        receipts.add(evidence_id)
        condition_id, role = allocation["amount_condition_id"], allocation["amount_role"]
        condition = conditions[condition_id]
        if role not in condition.roles or role in condition.derived_roles:
            raise ValueError("allocation needs a primitive actual receipt amount role")
        observations = [o for o in case.operands if o.condition_id == condition_id
                        and o.role == role and o.evidence_id == evidence_id and o.status == "obtained"]
        if not observations or any(o.value_type != "money" for o in observations):
            raise ValueError("allocation amount must be an actual typed money read")
        expression = MoneyExpression.model_validate({"bindings": {"receipt": {
            "kind": "operand", "condition_id": condition_id, "role": role,
            "evidence_id": evidence_id}}, "expression": {"op": "VALUE", "source": "receipt"}})
        value = BoundMoneyExpression(case, expression).evaluate()
        if value.currency != case.application.currency:
            raise ValueError("allocation currency must be independently confirmed by the receipt")
        if not allocation["condition_ids"] or condition_id not in allocation["condition_ids"]:
            raise ValueError("allocation must include its amount-source condition")
        for key in allocation["condition_ids"]:
            if key not in conditions or key in assigned:
                raise ValueError("each reviewed condition must target exactly one amount unit")
            assigned[key] = object_id
        units.append({"object_id": object_id, "scope": "item",
                      "claimed_amount": format(value.amount, "f"), "currency": value.currency,
                      "material_supported_max": None, "limit_status": "unknown", "limit": None})
        records.append({"object_id": object_id, "evidence_id": evidence_id,
                        "amount": format(value.amount, "f"), "currency": value.currency,
                        "refs": [ref.model_dump() for ref in value.refs],
                        "source_ids": sorted(value.sources)})
    if set(assigned) != set(conditions):
        raise ValueError("allocation cannot drop unassigned policy conditions")
    if sum((Decimal(unit["claimed_amount"]) for unit in units), Decimal(0)) != Decimal(case.application.claimed_amount):
        raise ValueError("source receipt amounts do not exactly account for the claim")
    raw = case.model_dump()
    for condition in raw["conditions"]:
        condition["object_id"] = assigned[condition["condition_id"]]
        condition["scope"] = "item"
    raw["amount_units"] = units
    raw["input_version"] = "patent_reader_v1:receipt_allocation:" + hashlib.sha256(
        (case.digest() + json.dumps(allocations, sort_keys=True)).encode()).hexdigest()
    allocated = AuditCase.model_validate(raw)
    return allocated, {"source_case_sha256": case.digest(), "allocated_case_sha256": allocated.digest(),
                       "source_receipts_exactly_partition_claim": True,
                       "legal_caps_or_eligibility_inferred_from_allocation": False, "units": records}
