"""Actual X/E/P -> relation -> file masking -> references -> correction -> F/B/T."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import torch
from torch import Tensor

from audit_engine.contracts import AuditCase
from audit_engine.inference.trace import AuditTrace
from audit_engine.models.heads import AuditHeads
from audit_engine.policy.amounts import project_amounts
from audit_engine.policy.comparators import normalize_text
from audit_engine.policy.programs import BoundMoneyExpression, compute_amount_units
from audit_engine.readers.value_formats import (
    declared_categories,
    explicit_printed_date_absence,
    missing_required_field,
)
from audit_engine.training.forward import (
    ScalarCondition,
    correction_inputs,
    forward_condition,
)

STATES = ("satisfied", "failed", "insufficient", "contradictory")
DEFAULT_THRESHOLDS = {
    "reference": 0.6,
    "support": 0.7,
    "conflict": 0.7,
    "negative_contribution": -0.05,
    "missed_contribution": 0.15,
}


@dataclass
class CaseFeatures:
    """Independent backbone encodings; no labels, targets, or answer features."""

    case_sha256: str
    evidence_ids: list[str]
    condition_ids: list[str]
    evidence: Tensor
    application: Tensor
    policies: Tensor
    signature: dict

    def validate(self, case: AuditCase, hidden_size: int):
        if self.case_sha256 != case.digest():
            raise ValueError(
                "feature cache was generated for a different input version"
            )
        if self.evidence_ids != [e.evidence_id for e in case.evidence]:
            raise ValueError("feature evidence ordering differs from runtime case")
        if self.condition_ids != [c.condition_id for c in case.conditions]:
            raise ValueError("feature condition ordering differs from runtime case")
        shapes = [
            (self.evidence, (len(case.evidence), hidden_size)),
            (self.application, (hidden_size,)),
            (self.policies, (len(case.conditions), hidden_size)),
        ]
        if any(
            tuple(t.shape) != shape or not bool(torch.isfinite(t).all())
            for t, shape in shapes
        ):
            raise ValueError("invalid or nonfinite feature matrix")
        if self.signature.get("pooling") != "last_non_padding_input_v1":
            raise ValueError("unexpected pooling contract")


def scalar_policies(case: AuditCase) -> list[ScalarCondition]:
    def canonical(operand, normalization):
        value = operand.value
        if operand.value_type in {"decimal", "money"} and isinstance(value, str):
            exact = Decimal(value)
            text = format(exact, "f")
            return (
                "0"
                if exact == 0
                else text.rstrip("0").rstrip(".")
                if "." in text
                else text
            )
        if isinstance(value, list):
            return (
                tuple(sorted({normalize_text(v, normalization) for v in value}))
                if operand.value_type == "set"
                else tuple(value)
            )
        if isinstance(value, str) and operand.value_type == "text":
            return normalize_text(value, normalization)
        return value

    policies = []
    evidence_ids = [e.evidence_id for e in case.evidence]
    for condition in case.conditions:
        operation = condition.operation
        values = []
        dependencies = []
        currencies = []
        dependency_rules = []
        for role in condition.roles:
            categories = declared_categories(condition.role_definitions.get(role, ""))
            row = []
            deps = []
            currency_row = []
            rule_row = []
            for evidence_id in evidence_ids:
                matches = [
                    o
                    for o in case.operands
                    if o.condition_id == condition.condition_id
                    and o.role == role
                    and o.evidence_id == evidence_id
                ]
                eligible_matches = [
                    o
                    for o in matches
                    if o.status == "obtained"
                    and (not categories or o.value in categories)
                    and not (o.value == "complete" and missing_required_field(
                        " ".join(ref.note for ref in o.refs), condition.role_definitions.get(role, "")))
                    and not (
                        operation
                        and operation.value_type == "money"
                        and role in {operation.left_role, operation.right_role}
                        and not o.currency
                    )
                ]
                obtained = [
                    canonical(o, operation.text_normalization if operation else "exact")
                    for o in eligible_matches
                ]
                # Multiple incompatible reads from a file remain ambiguous.
                row.append(obtained[0] if len(set(obtained)) == 1 else None)
                units = {o.currency for o in eligible_matches if o.currency}
                currency_row.append(next(iter(units)) if len(units) == 1 else None)
                rule_row.append(
                    tuple(
                        tuple(
                            tuple(evidence_ids.index(i) for i in group)
                            for group in (
                                o.dependency_groups or [[d] for d in o.depends_on]
                            )
                        )
                        for o in eligible_matches
                    )
                )
                deps.append(
                    tuple(
                        sorted(
                            {
                                evidence_ids.index(d)
                                for o in matches
                                for d in o.depends_on
                            }
                        )
                    )
                )
            values.append(tuple(row))
            dependencies.append(tuple(deps))
            currencies.append(tuple(currency_row))
            dependency_rules.append(tuple(rule_row))
        constant = operation.right_constant if operation else None
        if isinstance(constant, list):
            constant = tuple(constant)

        def currency_for(role):
            currencies = {
                o.currency
                for o in case.operands
                if o.condition_id == condition.condition_id
                and o.role == role
                and o.status == "obtained"
                and o.currency
            }
            return next(iter(currencies)) if len(currencies) == 1 else None

        policies.append(
            ScalarCondition(
                operation=operation.name if operation else None,
                values=tuple(values),
                required_roles=tuple(range(len(condition.roles))),
                left_role=condition.roles.index(operation.left_role)
                if operation
                else 0,
                right_role=condition.roles.index(operation.right_role)
                if operation and operation.right_role
                else 0,
                policy_verified=condition.compile_status == "verified",
                value_type=operation.value_type if operation else None,
                text_normalization=operation.text_normalization
                if operation
                else "exact",
                left_currency=currency_for(operation.left_role) if operation else None,
                right_currency=(
                    operation.currency
                    if operation.right_from_policy
                    else currency_for(operation.right_role)
                )
                if operation
                else None,
                inclusive=operation.inclusive if operation else True,
                grace_days=operation.grace_days if operation else 0,
                right_from_policy=operation.right_from_policy if operation else False,
                right_constant=constant,
                comparison_required=operation is not None,
                dependencies=tuple(dependencies),
                currencies=tuple(currencies),
                dependency_rules=tuple(dependency_rules),
                derived_programs={
                    condition.roles.index(role): BoundMoneyExpression(case, expression)
                    for role, expression in condition.derived_roles.items()
                },
                policy_sources=tuple(
                    sorted(
                        {
                            evidence_ids.index(r.evidence_id)
                            for r in condition.policy_refs
                        }
                    )
                ),
            )
        )
    return policies


def tensors_for_case(model: AuditHeads, case: AuditCase, features: CaseFeatures):
    hidden = model.shared.evidence.in_features
    features.validate(case, hidden)
    device = next(model.parameters()).device
    count = len(case.conditions)
    evidence_raw = features.evidence.to(device).expand(count, -1, -1)
    app_raw = features.application.to(device).expand(count, -1)
    policies_raw = features.policies.to(device)
    evidence, condition = model.shared(evidence_raw, app_raw, policies_raw)
    active = torch.ones(count, len(case.evidence), dtype=torch.bool, device=device)
    policies = scalar_policies(case)
    return condition, evidence, policies, active


def condition_source_refs(case, condition):
    refs = [
        ref
        for operand in case.operands
        if operand.condition_id == condition.condition_id
        and operand.status == "obtained"
        for ref in operand.refs
    ]
    if condition.compile_status == "verified":
        refs.extend(condition.policy_refs)
    for expression in condition.derived_roles.values():
        try:
            refs.extend(BoundMoneyExpression(case, expression).evaluate().refs)
        except ValueError:
            pass
    return list({ref.model_dump_json(): ref for ref in refs}.values())


def citation_masks(case: AuditCase, device) -> Tensor:
    """Source existence/provider checks only; no gold polarity is consulted."""
    ids = [e.evidence_id for e in case.evidence]
    mask = torch.zeros(
        len(case.conditions), len(ids), 2, dtype=torch.bool, device=device
    )
    for i, c in enumerate(case.conditions):
        for ref in condition_source_refs(case, c):
            mask[i, ids.index(ref.evidence_id), :] = True
    return mask


class AuditRunner:
    def __init__(
        self,
        model: AuditHeads,
        *,
        thresholds: dict | None = None,
        max_rounds: int = 2,
        qualification: str = "engineering",
        ablation: str = "full",
    ):
        if not 0 <= max_rounds <= 2:
            raise ValueError(
                "default patent runner unfolds at most two correction rounds"
            )
        self.model = model.eval()
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.max_rounds = max_rounds
        self.qualification = qualification
        if ablation not in {"full", "no_policy_comparison", "no_contribution"}:
            raise ValueError("unknown component ablation")
        self.ablation = ablation

    @torch.no_grad()
    def run(self, case: AuditCase, features: CaseFeatures) -> dict:
        trace = AuditTrace()
        trace.add(
            "source_and_citation_validation",
            [case.workflow_id],
            [case.digest()],
            details={
                "evidence": len(case.evidence),
                "conditions": len(case.conditions),
                "document_absence_proofs": [proof.model_dump() for proof in case.document_absences],
            },
        )
        calls = {"relation": 0, "contribution": 0, "corrector": 0}
        handles = []
        for name in calls:

            def increment(_module, _inputs, _output, name=name):
                calls[name] += 1

            handles.append(getattr(self.model, name).register_forward_hook(increment))
        try:
            facts = []
            if case.conditions:
                condition, evidence, policies, active = tensors_for_case(
                    self.model, case, features
                )
                verified = citation_masks(case, active.device)
                last = [None] * len(case.conditions)
                unchanged = torch.zeros(
                    len(case.conditions), dtype=torch.long, device=active.device
                )
                output = None
                recompute = torch.ones(len(case.conditions), dtype=torch.bool, device=active.device)
                for round_index in range(self.max_rounds + 1):
                    indices = recompute.nonzero().flatten()
                    fresh = forward_condition(
                        self.model,
                        condition[indices],
                        evidence[indices],
                        [policies[i] for i in indices.tolist()],
                        active[indices],
                        ablation=self.ablation,
                    )
                    output = fresh if output is None else {
                        name: previous.index_copy(0, indices, fresh[name])
                        for name, previous in output.items()
                    }
                    if any(
                        not bool(torch.isfinite(output[name]).all())
                        for name in ("logits", "references", "strengths", "delta")
                    ):
                        raise ValueError("nonfinite audit component output")
                    errors, fact, slots, eligible = correction_inputs(
                        evidence,
                        output,
                        active,
                        policies,
                        self.thresholds,
                        reference_verified=verified,
                    )
                    refs = (
                        output["references"].sigmoid() >= self.thresholds["reference"]
                    ) & verified
                    signatures = []
                    for i, compiled in enumerate(case.conditions):
                        state_index = int(fact[i, -4:].argmax())
                        ambiguous = any(
                            len({v for v in values if v is not None}) > 1
                            for values in policies[i].values
                        )
                        if ambiguous:
                            state_index = 3
                            eligible[i] = False
                        if compiled.compile_status != "verified":
                            state_index = 2
                            eligible[i] = False
                        signature = (state_index, tuple(refs[i].flatten().tolist()))
                        signatures.append(signature)
                        unchanged[i] = unchanged[i] + 1 if signature == last[i] else 0
                        trace.add(
                            "relation_masking_reference" if bool(recompute[i]) else "reuse_condition_result",
                            [compiled.condition_id, *features.evidence_ids],
                            [f"F:{compiled.condition_id}:{round_index}"],
                            condition_id=compiled.condition_id,
                            round_index=round_index,
                            details={
                                "state": STATES[state_index],
                                "errors": errors[i].int().tolist(),
                                "source_mask_recomputations": 0
                                if self.ablation == "no_contribution" or not bool(recompute[i])
                                else len(case.evidence),
                            },
                        )
                    last = signatures
                    if round_index == self.max_rounds:
                        break
                    eligible &= unchanged < 2
                    if not bool(eligible.any()):
                        break
                    before = condition
                    condition = self.model.corrector(
                        condition, output["context"], errors, fact, slots, eligible
                    )
                    assert torch.equal(before[~eligible], condition[~eligible])
                    recompute = eligible.clone()
                    trace.add(
                        "local_correction",
                        [f"h:{round_index}"],
                        [f"h:{round_index + 1}"],
                        round_index=round_index + 1,
                        details={
                            "updated_conditions": [
                                case.conditions[i].condition_id
                                for i in eligible.nonzero().flatten().tolist()
                            ]
                        },
                    )
                for i, compiled in enumerate(case.conditions):
                    references = {"support": [], "conflict": []}
                    available_refs = condition_source_refs(case, compiled)
                    for j, material in enumerate(case.evidence):
                        source_refs = [
                            ref.model_dump()
                            for ref in available_refs
                            if ref.evidence_id == material.evidence_id
                        ]
                        for q, name in enumerate(["support", "conflict"]):
                            if bool(refs[i, j, q]):
                                references[name].extend(source_refs)
                    gaps = []
                    for role_index, role in enumerate(compiled.roles):
                        observations = [o for o in case.operands if o.condition_id == compiled.condition_id and o.role == role]
                        available = bool(output["role_map"][i, role_index].any())
                        values = {v for v in policies[i].values[role_index] if v is not None}
                        categories = declared_categories(compiled.role_definitions.get(role, ""))
                        invalid_category = categories and any(
                            o.status == "obtained" and o.value not in categories for o in observations)
                        inconsistent_fields = any(o.status == "obtained" and o.value == "complete"
                            and missing_required_field(" ".join(ref.note for ref in o.refs),
                                                       compiled.role_definitions.get(role, "")) for o in observations)
                        if len(values) > 1 or (not available and (invalid_category or inconsistent_fields or any(o.status == "ambiguous" for o in observations))):
                            gap_state = "ambiguous"
                        elif available:
                            continue
                        elif any(proof.condition_id == compiled.condition_id and proof.role == role
                                 for proof in case.document_absences):
                            gap_state = "confirmed_missing"
                        elif (f"confirmed_missing:{compiled.condition_id}:{role}" in case.unresolved
                              and any(o.value_type == "date" and o.status == "not_obtained"
                                      and any(ref.kind == "visual" and explicit_printed_date_absence(ref.note) for ref in o.refs)
                                      for o in observations)):
                            gap_state = "confirmed_missing"
                        else:
                            gap_state = "not_read"
                        gaps.append({"role": role, "status": gap_state,
                                     "source_ids": sorted({o.evidence_id for o in observations})})
                    facts.append(
                        {
                            "object_id": compiled.object_id,
                            "condition_id": compiled.condition_id,
                            "S": float(output["strengths"][i, 0]),
                            "C": float(output["strengths"][i, 1]),
                            "G": bool(fact[i, 2]),
                            "gaps": gaps,
                            "state": STATES[last[i][0]],
                            "references": references,
                            "errors": errors[i].int().tolist(),
                            "compile_status": compiled.compile_status,
                        }
                    )
        finally:
            for handle in handles:
                handle.remove()
        decisions = {unit.object_id: "accept" for unit in case.amount_units}
        review = []
        condition_lookup = {c.condition_id: c for c in case.conditions}
        for fact in facts:
            condition = condition_lookup[fact["condition_id"]]
            state = fact["state"]
            action = "no_action"
            if state == "failed":
                action = condition.consequence.on_failed
            elif state == "insufficient":
                action = condition.consequence.on_gap
                if (action in {"reject_application", "reject_object"}
                        and not any(gap["status"] == "confirmed_missing" for gap in fact["gaps"])):
                    action = "request_material"
            elif state == "contradictory":
                action = condition.consequence.on_contradictory
            affected = (
                list(decisions)
                if action == "reject_application"
                else [condition.object_id]
            )
            if action in {"reject_application", "reject_object"}:
                for key in affected:
                    decisions[key] = "reject"
            elif (
                action in {"manual_review", "request_material"}
                or state in {"insufficient", "contradictory"}
                or any(fact["errors"])
            ):
                if decisions[condition.object_id] != "reject":
                    decisions[condition.object_id] = "review"
                review.append(
                    {
                        "object_id": condition.object_id,
                        "condition_id": condition.condition_id,
                        "reason": action
                        if action != "no_action"
                        else "unresolved_module_errors",
                        "errors": fact["errors"],
                        "references": fact["references"],
                        "required_roles": condition.roles,
                        "required_material_types": [
                            {"role": gap["role"], "source_kind": condition.role_origins.get(gap["role"], "any"), "status": gap["status"]}
                            for gap in fact["gaps"]
                        ],
                    }
                )
        if case.policy_coverage != "verified" or not case.conditions or case.unresolved:
            for key in decisions:
                if decisions[key] != "reject":
                    decisions[key] = "review"
            review.append(
                {
                    "object_id": "application",
                    "reason": "policy_coverage_or_reader_unresolved",
                    "details": case.unresolved,
                }
            )
        amount_units, amount_issues, amount_traces = compute_amount_units(
            case, decisions
        )
        for issue in amount_issues:
            if decisions[issue["object_id"]] != "reject":
                decisions[issue["object_id"]] = "review"
            review.append(issue)
        if case.amount_programs:
            trace.add(
                "source_bound_amount_program",
                ["operands", "P"],
                ["amount_units"],
                details={"calculations": amount_traces, "unresolved": amount_issues},
            )
        amounts = project_amounts(
            amount_units,
            decisions,
            case.application.claimed_amount,
            case.application.currency,
        )
        for unit in amounts["units"]:
            if unit["state"] == "review" and not any(
                r["object_id"] == unit["object_id"] for r in review
            ):
                review.append(
                    {
                        "object_id": unit["object_id"],
                        "reason": "amount_basis_unresolved",
                    }
                )
        amount_lookup = {unit["object_id"]: unit for unit in amounts["units"]}
        for task in review:
            unit = amount_lookup.get(task["object_id"])
            task["review_amount"] = (
                amounts["A_review"] if task["object_id"] == "application"
                else unit["within_limit"] if unit and unit["state"] == "review" else "0"
            )
            task["currency"] = amounts["currency"]
            task.setdefault("references", {"support": [], "conflict": []})
            task.setdefault("required_material_types", [])
            task.setdefault("errors", [0, 0, 0])
        trace.add(
            "amount_projection",
            ["F", "P", "X"],
            ["B"],
            details={"conservation_valid": amounts["conservation_valid"]},
        )
        trace.add(
            "review_tasks", ["F", "B"], ["review"], details={"tasks": len(review)}
        )
        return {
            "schema_version": "patent-output-v1",
            "workflow_id": case.workflow_id,
            "case_sha256": case.digest(),
            "qualification": self.qualification,
            "ablation": self.ablation,
            "max_correction_rounds": self.max_rounds,
            "F": facts,
            "B": amounts,
            "T": trace.events,
            "review": review,
            "module_calls": calls,
            "answer": {
                "FLAG": int(
                    not review
                    and all(unit["state"] == "accept" for unit in amounts["units"])
                    and Decimal(amounts["A_review"]) == 0
                    and Decimal(amounts["A_ok"]) == Decimal(amounts["A_claim"])
                ),
                "max_legal": amounts["A_max"],
                "problem_reports": [
                    f"{f['condition_id']}: {f['state']}"
                    for f in facts
                    if f["state"] != "satisfied"
                ],
            },
        }
