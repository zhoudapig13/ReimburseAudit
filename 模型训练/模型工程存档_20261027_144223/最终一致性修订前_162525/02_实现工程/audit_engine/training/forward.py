"""Differentiable file masking with source-dependent policy recomputation.

The comparator interface supplies runtime operands, never training labels.
Typed comparisons and explicitly declared open-text conditions use one
unambiguous value per role. Complex policy combinations remain unresolved.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from audit_engine.models.heads import AuditHeads, aggregate_strengths
from audit_engine.policy.comparators import Value, evaluate
from audit_engine.policy.programs import BoundMoneyExpression


@dataclass(frozen=True)
class ScalarCondition:
    operation: str | None
    values: tuple[tuple[Value, ...], ...]
    required_roles: tuple[int, ...]
    left_role: int = 0
    right_role: int = 1
    policy_verified: bool = True
    value_type: str | None = None
    left_currency: str | None = None
    right_currency: str | None = None
    inclusive: bool = True
    grace_days: int = 0
    right_from_policy: bool = False
    right_constant: Value = None
    comparison_required: bool = True
    dependencies: tuple[tuple[tuple[int, ...], ...], ...] | None = None
    policy_sources: tuple[int, ...] = ()
    currencies: tuple[tuple[str | None, ...], ...] | None = None
    text_normalization: str = "exact"
    dependency_rules: (
        tuple[tuple[tuple[tuple[tuple[int, ...], ...], ...], ...], ...] | None
    ) = None
    derived_programs: dict[int, BoundMoneyExpression] | None = None

    def active_currency(self, role: int, active: Tensor, fallback: str | None):
        if self.derived_programs and role in self.derived_programs:
            try:
                return self.derived_programs[role].evaluate(active).currency
            except ValueError:
                return None
        if self.currencies is None:
            return fallback
        values = {
            self.currencies[role][index]
            for index, value in enumerate(self.values[role])
            if value is not None and self.source_active(role, index, active)
        }
        return next(iter(values)) if len(values) == 1 else None

    def source_active(self, role: int, index: int, active: Tensor) -> bool:
        if not bool(active[index]):
            return False
        if self.dependency_rules is not None:
            return any(
                all(any(bool(active[j]) for j in group) for group in rule)
                for rule in self.dependency_rules[role][index]
            )
        return self.dependencies is None or all(
            bool(active[j]) for j in self.dependencies[role][index]
        )

    def semantic_operands_ready(self, active: Tensor) -> bool:
        if self.comparison_required or not self.policy_verified:
            return False
        for role in self.required_roles:
            if self.derived_programs and role in self.derived_programs:
                try:
                    self.derived_programs[role].evaluate(active)
                    continue
                except ValueError:
                    return False
            available = {
                value
                for index, value in enumerate(self.values[role])
                if value is not None and self.source_active(role, index, active)
            }
            if len(available) != 1:
                return False
            value = next(iter(available))
            if not (
                (isinstance(value, str) and value.strip())
                or (
                    isinstance(value, tuple)
                    and value
                    and all(isinstance(v, str) and v.strip() for v in value)
                )
            ):
                return False
        return True

    def rebuild(self, active: Tensor, role_count: int) -> tuple[Tensor, ...]:
        """Discard masked sources before resolving values and comparing them."""
        device = active.device
        count = active.numel()
        role_map = torch.zeros(role_count, count, dtype=torch.bool, device=device)
        required = torch.zeros(role_count, dtype=torch.bool, device=device)
        if (
            not self.required_roles
            or len(self.values) > role_count
            or any(role < 0 or role >= len(self.values) for role in self.required_roles)
            or (self.comparison_required and self.left_role not in self.required_roles)
            or (
                self.comparison_required
                and not self.right_from_policy
                and self.right_role not in self.required_roles
            )
        ):
            raise ValueError("invalid policy roles")
        if not self.comparison_required and (
            self.operation is not None or self.right_from_policy
        ):
            raise ValueError(
                "open-text conditions must explicitly omit deterministic comparison"
            )
        required[list(self.required_roles)] = True
        resolved = []
        for role, candidates in enumerate(self.values):
            if self.derived_programs and role in self.derived_programs:
                try:
                    value = self.derived_programs[role].evaluate(active)
                    resolved.append(format(value.amount, "f"))
                    for index, identifier in enumerate(
                        self.derived_programs[role].evidence_ids
                    ):
                        if identifier in value.sources:
                            role_map[role, index] = True
                except ValueError:
                    resolved.append(None)
                continue
            if len(candidates) != count:
                raise ValueError("operand source count differs from evidence count")
            available = set()
            for index, value in enumerate(candidates):
                if value is not None and self.source_active(role, index, active):
                    role_map[role, index] = True
                    available.add(value)
            resolved.append(next(iter(available)) if len(available) == 1 else None)
        complete = (
            self.policy_verified
            and all(bool(active[j]) for j in self.policy_sources)
            and all(resolved[role] is not None for role in self.required_roles)
        )
        result = 2
        if complete and self.comparison_required:
            left = resolved[self.left_role]
            if self.operation == "CLOSED_ROUTE":
                left = tuple(resolved[role] for role in self.required_roles if role != self.right_role)
            right = (
                self.right_constant
                if self.right_from_policy
                else resolved[self.right_role]
            )
            outcome = evaluate(
                self.operation,
                left,
                right,
                value_type=self.value_type,
                left_currency=self.active_currency(
                    self.left_role, active, self.left_currency
                ),
                right_currency=self.right_currency
                if self.right_from_policy
                else self.active_currency(self.right_role, active, self.right_currency),
                inclusive=self.inclusive,
                grace_days=self.grace_days,
                text_normalization=self.text_normalization,
            )
            complete = outcome.complete
            if complete:
                result = 0 if outcome.passed else 1
        elif not self.comparison_required:
            complete = False
        comparison = torch.zeros(count, 3, device=device)
        comparison[:, 2] = 1
        mask = torch.zeros(count, 1, device=device)
        if complete:
            providers = role_map[list(self.required_roles)].any(dim=0)
            comparison[providers] = 0
            comparison[providers, result] = 1
            mask[providers] = 1
        return role_map, required, comparison, mask


def forward_condition(
    model: AuditHeads,
    condition: Tensor,
    evidence: Tensor,
    policies: list[ScalarCondition],
    active: Tensor,
    *,
    ablation: str = "full",
) -> dict[str, Tensor]:
    """Recompute all affected role contexts and rules for every file removal."""
    if ablation not in {"full", "no_policy_comparison", "no_contribution"}:
        raise ValueError("unknown component ablation")
    roles = model.relation.context.projection.in_features // (evidence.size(-1) + 1)
    rebuilt = [p.rebuild(a, roles) for p, a in zip(policies, active, strict=True)]
    role_map, required, comparison, complete = (
        torch.stack([row[index] for row in rebuilt]) for index in range(4)
    )
    if ablation == "no_policy_comparison":
        comparison = torch.zeros_like(comparison)
        comparison[..., 2] = 1
        complete = torch.zeros_like(complete)
    context = model.relation.context(evidence, role_map, active)
    fused, logits, probabilities = model.relation(
        condition, evidence, context, comparison, complete
    )
    strengths = aggregate_strengths(probabilities, role_map, active, required)
    contributions = []
    for index in range(0 if ablation == "no_contribution" else evidence.size(1)):
        masked_active = active.clone()
        masked_active[:, index] = False
        masked = [
            p.rebuild(a, roles) for p, a in zip(policies, masked_active, strict=True)
        ]
        masked_roles, _, masked_comparison, masked_complete = (
            torch.stack([row[j] for row in masked]) for j in range(4)
        )
        if ablation == "no_policy_comparison":
            masked_comparison = torch.zeros_like(masked_comparison)
            masked_comparison[..., 2] = 1
            masked_complete = torch.zeros_like(masked_complete)
        masked_context = model.relation.context(evidence, masked_roles, masked_active)
        _, _, masked_probabilities = model.relation(
            condition, evidence, masked_context, masked_comparison, masked_complete
        )
        masked_strengths = aggregate_strengths(
            masked_probabilities, masked_roles, masked_active, required
        )
        contributions.append((strengths - masked_strengths) * active[:, index, None])
    if ablation == "no_contribution":
        delta = torch.zeros_like(probabilities[..., :2])
        references = torch.logit(probabilities[..., :2].clamp(1e-6, 1 - 1e-6))
    else:
        delta = torch.stack(contributions, dim=1)
        references = model.contribution(fused, probabilities, delta)
    return {
        "logits": logits,
        "probabilities": probabilities,
        "references": references,
        "strengths": strengths,
        "delta": delta,
        "context": context,
        "role_map": role_map,
        "required": required,
        "comparison": comparison,
        "complete": complete,
    }


def correction_inputs(
    evidence: Tensor,
    output: dict[str, Tensor],
    active: Tensor,
    policies: list[ScalarCondition],
    thresholds: dict[str, float] | None = None,
    reference_verified: Tensor | None = None,
    upstream_read_failure: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build runtime reference/relation/dependence signals without any targets.

    Scalar fixture providers are source-verified by the caller. A document
    reader must additionally verify quotes before using this for real inputs.
    """
    limits = thresholds or {"reference": 0.6, "support": 0.7, "conflict": 0.7}
    probabilities = output["probabilities"]
    role_map, required = output["role_map"], output["required"]
    comparison, complete = output["comparison"], output["complete"]
    ref = output["references"].detach().sigmoid() >= limits["reference"]
    if reference_verified is not None:
        if reference_verified.shape != ref.shape:
            raise ValueError("reference verification mask has incompatible shape")
        ref &= reference_verified.bool()
    gap = ~(role_map.any(dim=-1) | ~required).all(dim=-1)
    comparison_ready = (complete.squeeze(-1).bool() | ~role_map.any(dim=1)).all(dim=-1)
    needs_comparison = torch.tensor(
        [p.comparison_required for p in policies],
        dtype=torch.bool,
        device=active.device,
    )
    # A present number without confirmed currency, date interpretation, or an
    # executable rule is still an information gap for deterministic conditions.
    gap |= needs_comparison & ~comparison_ready
    # Only complete, source-backed deterministic comparisons identify polarity.
    expected = comparison.argmax(dim=-1)
    current = probabilities.detach().argmax(dim=-1)
    mismatch = (current != expected) & complete.squeeze(-1).bool() & active
    reference_polarity = expected.clone()
    for index, policy in enumerate(policies):
        if not policy.comparison_required:
            reference_polarity[index] = current[index]
    same_polarity_ref = ref.gather(
        2, reference_polarity.clamp_max(1)[..., None]
    ).squeeze(-1)
    same_polarity_ref &= reference_polarity != 2
    coverage = (role_map & same_polarity_ref[:, None, :]).any(dim=-1)
    ref_error_roles = required & role_map.any(dim=-1) & ~coverage
    v_ref = ref_error_roles.any(dim=-1)
    # Equivalent scalar providers require role, polarity and identical value.
    equivalent_coverage = torch.zeros_like(ref)
    for row, policy in enumerate(policies):
        for index in range(active.size(1)):
            provider_roles = role_map[row, :, index].nonzero().flatten().tolist()
            if provider_roles:
                for polarity in range(2):
                    equivalent_coverage[row, index, polarity] = all(
                        any(
                            bool(role_map[row, role, other])
                            and bool(ref[row, other, polarity])
                            and policy.values[role][other] == policy.values[role][index]
                            and int(reference_polarity[row, other]) == polarity
                            for other in range(active.size(1))
                        )
                        for role in provider_roles
                    )
    abnormal = (
        ref & (output["delta"].detach() < limits.get("negative_contribution", -0.05))
    ).any(dim=-1)
    missed = (
        ~ref
        & (output["delta"].detach() > limits.get("missed_contribution", 0.15))
        & ~equivalent_coverage
    ).any(dim=-1)
    dep_error = (abnormal | missed) & active
    errors = torch.stack([v_ref, mismatch.any(-1), dep_error.any(-1)], dim=-1).float()
    safe = torch.tensor(
        [p.policy_verified for p in policies], dtype=torch.bool, device=active.device
    ) & (complete.squeeze(-1).bool() | ~role_map.any(dim=1)).all(dim=-1)
    for index, policy in enumerate(policies):
        if not policy.comparison_required:
            safe[index] = policy.semantic_operands_ready(active[index])
    eligible = errors.bool().any(-1) & ~gap & safe
    if upstream_read_failure is not None:
        if upstream_read_failure.shape != eligible.shape:
            raise ValueError("upstream reading failure mask has incompatible shape")
        eligible &= ~upstream_read_failure.bool()
    addressed = role_map & (mismatch | dep_error)[:, None, :]
    addressed |= role_map & ref_error_roles[:, :, None]
    weights = addressed.to(evidence.dtype)
    counts = weights.sum(-1, keepdim=True)
    means = torch.einsum("brn,bnd->brd", weights, evidence) / counts.clamp_min(1)
    comparisons = torch.einsum("brn,bnc->brc", weights, comparison) / counts.clamp_min(
        1
    )
    slots = torch.cat([means, comparisons, (counts > 0).float()], dim=-1)
    support, conflict = output["strengths"].unbind(-1)
    states = torch.full_like(support, 2, dtype=torch.long)
    s, c = (
        support.detach() >= limits["support"],
        conflict.detach() >= limits["conflict"],
    )
    states[s & ~c] = 0
    states[c & ~s] = 1
    states[s & c] = 3
    states[gap] = 2
    fact = torch.cat(
        [
            output["strengths"],
            gap[:, None].float(),
            torch.nn.functional.one_hot(states, 4).float(),
        ],
        dim=-1,
    )
    return errors, fact, slots, eligible
