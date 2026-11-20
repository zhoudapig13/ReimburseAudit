"""Compile reviewed attendance-based allowance scope into source-bound programs."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from audit_engine.contracts import AuditCase, Condition, PolicyNumber
from audit_engine.policy.programs import BoundMoneyExpression
from audit_engine.readers.text_fields import read_original_text_field


def compile_attendance_allowance(case: AuditCase, profile: dict, source_root: Path):
    """A reviewed profile supplies locators and scope, never expected outcomes.

    Numeric rates/caps come from already verified policy conditions. Contract
    dates and worker IDs are parsed from original DOCX fields. The daily ledger
    and signed per-worker acknowledgements must agree for every worker.
    """
    if profile["workflow_id"] != case.workflow_id:
        raise ValueError("attendance profile belongs to a different workflow")
    if not profile.get("reviewer") or not profile.get("scope_review"):
        raise ValueError("attendance policy scope requires a recorded review")
    sources = {e.evidence_id: e for e in case.evidence}
    for key, expected in profile["source_sha256"].items():
        material = sources[key]
        path = (source_root / material.raw_ref).resolve()
        if (not path.is_relative_to(source_root.resolve()) or expected != material.source_sha256
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise ValueError("attendance scope source version changed")
    conditions = {c.condition_id: c for c in case.conditions}
    cap, rate = (conditions[profile[key]] for key in ("cap_condition", "rate_condition"))
    if any(c.compile_status != "verified" or not c.operation
           or not c.operation.right_from_policy or c.operation.value_type != "money"
           for c in (cap, rate)):
        raise ValueError("allowance profile requires verified monetary cap and rate")
    if rate.operation.currency != cap.operation.currency:
        raise ValueError("allowance rate and cap currencies differ")
    identifiers = {name: profile["prefix"] + "_" + name for name in (
        "attendance_basis", "vat_invoice", "service_start", "service_end")}
    if any(key in conditions for key in identifiers.values()):
        raise ValueError("attendance profile already compiled")
    contract, invoice = profile["contract"], profile["invoice"]
    period_role = "approved_" + contract + "_contract_period"
    workers_role = "approved_" + contract + "_worker_ids"
    invoice_role = cap.operation.left_role
    calculated_role = "calculated_attested_allowance"
    definition = {
        period_role: "Read the start and end of the contract Duration field as one date interval.",
        workers_role: "Read the complete ID Number roster from the original subcontract.",
    }
    segment_lookup = {s.segment_id: (e.evidence_id, s.text) for e in case.evidence for s in e.segments}

    def refs(keys):
        return [{"evidence_id": segment_lookup[key][0], "segment_id": key,
                 "quote": segment_lookup[key][1]} for key in keys]

    review_digest = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
    method = "source_contract_attendance_signature_scope_v1:" + review_digest
    base = {"object_id": cap.object_id, "scope": "application", "compile_status": "verified",
            "verification_method": method,
            "consequence": {"on_failed": "request_material", "on_gap": "request_material"}}
    condition_id = identifiers["attendance_basis"]
    bindings = {
        "days": {"kind": "table_sum", **profile["attendance"],
                 "period_condition_id": condition_id, "period_role": period_role,
                 "members_condition_id": condition_id, "members_role": workers_role},
        "signed_days": {"kind": "table_sum", **profile["signatures"],
                        "members_condition_id": condition_id, "members_role": workers_role},
        "daily_rate": {"kind": "policy_number", "parameter_id": profile["prefix"] + "_daily_rate"},
        "cap": {"kind": "policy_number", "parameter_id": profile["prefix"] + "_cap"},
    }
    limit = {"bindings": bindings, "expression": {"op": "MIN", "args": [
        {"op": "VALUE", "source": "daily_rate", "quantity_source": "days"},
        {"op": "VALUE", "source": "cap"}]}, "table_agreements": [["days", "signed_days"]]}
    additions = [Condition.model_validate({**base, "condition_id": condition_id,
        "question": "Is the invoice total supported by authorized in-period worker-days, signed worker acknowledgements and the contractual cap?",
        "kind": "amount", "roles": [invoice_role, calculated_role, period_role, workers_role],
        "role_types": {invoice_role: "money", period_role: "interval", workers_role: "set"},
        "role_origins": {invoice_role: "receipt", period_role: "background", workers_role: "background"},
        "role_definitions": definition,
        "policy_refs": refs(profile["attendance_policy_refs"]),
        "operation": {"name": "LE", "left_role": invoice_role, "right_role": calculated_role,
                      "value_type": "money", "currency": rate.operation.currency},
        "derived_roles": {calculated_role: limit}})]
    vat_role = "actual_" + invoice + "_vat_invoice_type"
    additions.append(Condition.model_validate({**base, "condition_id": identifiers["vat_invoice"],
        "question": "Is the original allowance document an official general or special VAT invoice?",
        "kind": "required", "roles": [vat_role], "role_origins": {vat_role: "receipt"},
        "role_definitions": {vat_role: "Classify the actual printed invoice header as general_vat_invoice, special_vat_invoice, or other_document. Quote or transcribe the visible document header; do not infer validity from the claim."},
        "role_types": {vat_role: "text"}, "policy_refs": refs(profile["invoice_policy_refs"]),
        "operation": {"name": "IN_SET", "left_role": vat_role,
                      "right_constant": ["general_vat_invoice", "special_vat_invoice"],
                      "right_from_policy": True, "value_type": "text"}}))
    for boundary in ("start", "end"):
        role = "actual_" + invoice + "_service_" + boundary + "_date"
        additions.append(Condition.model_validate({**base, "condition_id": identifiers["service_" + boundary],
            "question": "Does the allowance service " + boundary + " date stated on the original invoice fall within the contract period?",
            "kind": "required", "roles": [role, period_role],
            "role_origins": {role: "receipt", period_role: "background"},
            "role_types": {role: "date", period_role: "interval"},
            "role_definitions": {period_role: definition[period_role], role:
                "Read the " + boundary + " calendar date of the allowance WORK/SERVICE period in the invoice Remarks, not the invoice issue date or payment date."},
            "policy_refs": refs(profile["attendance_policy_refs"]),
            "operation": {"name": "WITHIN", "left_role": role, "right_role": period_role, "value_type": "date"}}))
    raw = case.model_dump()
    raw["conditions"].extend(c.model_dump() for c in additions)
    for name, policy in (("daily_rate", rate), ("cap", cap)):
        raw["policy_numbers"].append(PolicyNumber.model_validate({
            "parameter_id": profile["prefix"] + "_" + name,
            "value": policy.operation.right_constant, "currency": policy.operation.currency,
            "refs": [r.model_dump() for r in policy.policy_refs],
            "compile_status": "verified", "verification_method": method}).model_dump())
    for old in case.operands:
        if old.condition_id == cap.condition_id and old.role == invoice_role:
            observation = old.model_dump()
            observation["condition_id"] = condition_id
            raw["operands"].append(observation)
    compiled = AuditCase.model_validate(raw)
    field_reads = []
    for target in [condition_id, identifiers["service_start"], identifiers["service_end"]]:
        observation, record = read_original_text_field(compiled, source_root,
            evidence_id=contract, condition_id=target, role=period_role,
            mode="labeled_date_interval", label=profile["contract_period_label"])
        raw["operands"].append(observation.model_dump())
        field_reads.append(record)
    observation, record = read_original_text_field(compiled, source_root,
        evidence_id=contract, condition_id=condition_id, role=workers_role,
        mode="labeled_identifier_set", label=profile["worker_identifier_label"])
    raw["operands"].append(observation.model_dump())
    field_reads.append(record)
    for program in raw["amount_programs"]:
        if program["object_id"] == cap.object_id:
            program["limit"] = copy.deepcopy(limit)
            program["verification_method"] = method
    raw["policy_coverage"] = "verified"
    raw["unresolved"] = [v for v in raw["unresolved"] if v != "remaining_policy_coverage_requires_review"]
    raw["input_version"] = "patent_reader_v1:attested_allowance_scope:" + review_digest
    result = AuditCase.model_validate(raw)
    # Evaluate source arithmetic now; neural decisions and invoice reads remain
    # separate and still run through the full three-component pipeline.
    attendance_condition = next(c for c in result.conditions if c.condition_id == condition_id)
    quantity = BoundMoneyExpression(result, attendance_condition.derived_roles[calculated_role]).evaluate()
    return result, {"profile_sha256": review_digest, "scope_review": profile["scope_review"],
        "field_reads": field_reads, "source_calculated_limit": str(quantity.amount),
        "source_ids": sorted(quantity.sources), "targets_read_by_compiler": False,
        "pending_invoice_roles": [vat_role, "actual_" + invoice + "_service_start_date",
                                  "actual_" + invoice + "_service_end_date"]}
