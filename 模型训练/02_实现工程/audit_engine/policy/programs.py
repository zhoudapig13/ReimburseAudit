"""Source-bound money expressions shared by comparisons and final amounts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, localcontext

from audit_engine.contracts import AuditCase, Citation, MoneyExpression
from audit_engine.policy.table_numbers import sum_source_table
from audit_engine.readers.value_formats import date_value


@dataclass(frozen=True)
class MoneyValue:
    amount: Decimal
    currency: str | None
    sources: frozenset[str]
    refs: tuple[Citation, ...]


def number(value, *, source=True):
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError("invalid source numeric value") from None
    if (
        not result.is_finite()
        or result < 0
        or result.adjusted() > 18
        or (source and result.as_tuple().exponent < -12)
    ):
        raise ValueError("source numeric value is outside the supported range")
    return result


def operand_available(operand, active):
    groups = operand.dependency_groups or [
        [identifier] for identifier in operand.depends_on
    ]
    return (
        operand.status == "obtained"
        and operand.evidence_id in active
        and all(any(identifier in active for identifier in group) for group in groups)
    )


class BoundMoneyExpression:
    def __init__(self, case: AuditCase, expression: MoneyExpression):
        self.case = case
        self.expression = expression
        self.evidence_ids = [e.evidence_id for e in case.evidence]

    def evaluate(self, active=None):
        available = (
            set(self.evidence_ids)
            if active is None
            else {
                identifier
                for identifier, enabled in zip(self.evidence_ids, active, strict=True)
                if bool(enabled)
            }
        )
        values = {}
        member_sets = {}
        table_groups = {}
        for key, binding in self.expression.bindings.items():
            if binding.kind == "table_sum":
                if binding.evidence_id not in available:
                    raise ValueError("source table is masked")
                material = next(e for e in self.case.evidence if e.evidence_id == binding.evidence_id)
                period, context_sources, context_refs = None, set(), []
                if binding.period_role:
                    observations = [o for o in self.case.operands if o.condition_id == binding.period_condition_id
                                    and o.role == binding.period_role and operand_available(o, available)]
                    periods = {tuple(o.value) for o in observations if o.value_type == "interval" and isinstance(o.value, list) and len(o.value) == 2}
                    if len(periods) != 1:
                        raise ValueError("table scope has no unambiguous source-bound date interval")
                    normalized = [date_value(value) for value in next(iter(periods))]
                    if not all(normalized):
                        raise ValueError("table scope date interval remains ambiguous")
                    period = tuple(date.fromisoformat(value) for value in normalized)
                    if period[0] > period[1]:
                        raise ValueError("table scope date interval is reversed")
                    context_sources = {identifier for o in observations for identifier in o.depends_on if identifier in available}
                    context_refs = [ref for o in observations for ref in o.refs]
                amount, grouped, refs = sum_source_table(material, binding, period)
                if binding.members_role:
                    member_observations = [o for o in self.case.operands if o.condition_id == binding.members_condition_id
                                           and o.role == binding.members_role and operand_available(o, available)]
                    member_sets_read = {tuple(sorted(set(o.value))) for o in member_observations if o.value_type == "set" and isinstance(o.value, list)}
                    if len(member_sets_read) != 1:
                        raise ValueError("table has no unambiguous authorized member set")
                    if not set(grouped).issubset(set(next(iter(member_sets_read)))):
                        raise ValueError("table contains members outside the authorized source roster")
                    context_sources.update(identifier for o in member_observations for identifier in o.depends_on if identifier in available)
                    context_refs.extend(ref for o in member_observations for ref in o.refs)
                values[key] = MoneyValue(number(amount), None, frozenset({binding.evidence_id} | context_sources), refs + tuple(context_refs))
                table_groups[key] = grouped
                continue
            if binding.kind == "policy_number":
                parameter = next(
                    p
                    for p in self.case.policy_numbers
                    if p.parameter_id == binding.parameter_id
                )
                sources = {r.evidence_id for r in parameter.refs}
                if parameter.compile_status != "verified" or not sources.issubset(
                    available
                ):
                    raise ValueError("numeric policy source is unresolved or masked")
                values[key] = MoneyValue(
                    number(parameter.value),
                    parameter.currency,
                    frozenset(sources),
                    tuple(parameter.refs),
                )
                continue
            observations = [
                o
                for o in self.case.operands
                if o.condition_id == binding.condition_id
                and o.role == binding.role
                and (
                    binding.evidence_id is None or o.evidence_id == binding.evidence_id
                )
                and operand_available(o, available)
                and (o.value_type != "money" or (
                    isinstance(o.currency, str) and len(o.currency) == 3 and o.currency.isupper()
                ))
            ]
            if not observations:
                raise ValueError(
                    "required amount source was not obtained or was masked"
                )
            condition = next(
                c
                for c in self.case.conditions
                if c.condition_id == binding.condition_id
            )
            if condition.compile_status != "verified":
                raise ValueError("amount source role definition is not verified")
            canonical = {
                (
                    tuple(sorted(set(o.value)))
                    if isinstance(o.value, list)
                    else number(o.value)
                    if o.value_type in {"money", "decimal"}
                    else o.value,
                    o.currency,
                    "numeric" if o.value_type in {"money", "decimal"} else o.value_type,
                )
                for o in observations
            }
            if len(canonical) != 1:
                raise ValueError("amount source observations are ambiguous")
            value, currency, value_type = next(iter(canonical))
            if binding.kind == "count_members":
                if not isinstance(value, tuple) or value_type != "set":
                    raise ValueError(
                        "member count requires an explicitly read set of members"
                    )
                members = set(value)
                all_members = members.copy()
                if binding.match_suffix is not None:
                    suffix = " ".join(binding.match_suffix.casefold().split())
                    members = {
                        v
                        for v in members
                        if " ".join(v.casefold().split()).endswith(suffix)
                    }
                amount, currency = Decimal(len(members)), None
                member_sets[key] = (all_members, members)
            else:
                if value_type != "numeric":
                    raise ValueError("amount binding requires a typed primitive number")
                if any(o.value_type == "money" for o in observations) and (
                    not currency or len(currency) != 3 or not currency.isupper()
                ):
                    raise ValueError("source money currency is not confirmed")
                amount = number(value)
            values[key] = MoneyValue(
                amount,
                currency,
                frozenset(
                    identifier
                    for o in observations
                    for identifier in o.depends_on
                    if identifier in available
                ),
                tuple(ref for o in observations for ref in o.refs),
            )
        for partition in self.expression.member_partitions:
            original = member_sets[partition[0]][0]
            selected = [member_sets[key][1] for key in partition]
            if set().union(*selected) != original or sum(map(len, selected)) != len(
                original
            ):
                raise ValueError(
                    "participant classifications are incomplete or overlap"
                )
        for agreement in self.expression.table_agreements:
            original = table_groups[agreement[0]]
            if any(table_groups[key] != original for key in agreement[1:]):
                raise ValueError("grouped source attendance and signed totals disagree")
        budget = [128]

        def visit(node, depth=0):
            budget[0] -= 1
            if depth > 16 or budget[0] < 0 or not isinstance(node, dict):
                raise ValueError("amount expression exceeds structural budget")
            op = node.get("op")
            if op == "VALUE":
                if (
                    not set(node).issubset({"op", "source", "quantity_source"})
                    or node.get("source") not in values
                ):
                    raise ValueError("VALUE needs a declared source binding")
                value = values[node["source"]]
                if "quantity_source" in node:
                    quantity = values.get(node["quantity_source"])
                    if quantity is None or quantity.currency is not None:
                        raise ValueError(
                            "quantity must be an independently sourced dimensionless number"
                        )
                    return MoneyValue(
                        number(value.amount * quantity.amount, source=False),
                        value.currency,
                        value.sources | quantity.sources,
                        value.refs + quantity.refs,
                    )
                return value
            expected = (
                {"op", "args", "from_currency", "to_currency", "mode"}
                if op == "CONVERT"
                else {"op", "args"}
            )
            if set(node) != expected or not isinstance(node.get("args"), list):
                raise ValueError("invalid amount expression shape")
            args = [visit(child, depth + 1) for child in node["args"]]
            if not args:
                raise ValueError("amount operation requires arguments")
            sources = frozenset().union(*(a.sources for a in args))
            refs = tuple(ref for a in args for ref in a.refs)
            currency = args[0].currency
            if op == "CONVERT":
                if (
                    len(args) != 2
                    or currency != node["from_currency"]
                    or args[1].currency is not None
                    or node["mode"] not in {"multiply", "divide"}
                    or not isinstance(node["to_currency"], str)
                    or len(node["to_currency"]) != 3
                    or not node["to_currency"].isupper()
                    or args[1].amount == 0
                ):
                    raise ValueError(
                        "conversion requires a source currency, target currency and positive sourced rate"
                    )
                result = (
                    args[0].amount * args[1].amount
                    if node["mode"] == "multiply"
                    else args[0].amount / args[1].amount
                )
                return MoneyValue(
                    number(result, source=False), node["to_currency"], sources, refs
                )
            if any(a.currency != currency for a in args):
                raise ValueError(
                    "amount operation mixes currencies without explicit conversion"
                )
            if op == "SUM":
                result = sum((a.amount for a in args), Decimal(0))
            elif op == "MIN":
                result = min(a.amount for a in args)
            elif op == "CAP" and len(args) == 2:
                result = min(a.amount for a in args)
            elif op == "EXCLUDE" and len(args) == 2:
                result = max(Decimal(0), args[0].amount - args[1].amount)
            else:
                raise ValueError("amount operation or arity is outside the whitelist")
            return MoneyValue(number(result, source=False), currency, sources, refs)

        with localcontext() as context:
            context.prec = 50
            result = visit(self.expression.expression)
        # Agreement/partition bindings can be essential even when they are not
        # numeric terms in the final expression. Keep those validation sources
        # in role maps, citations and counterfactual masking dependencies.
        constraint_keys = {key for group in (
            self.expression.table_agreements + self.expression.member_partitions
        ) for key in group}
        sources = result.sources | frozenset(
            identifier for key in constraint_keys for identifier in values[key].sources)
        refs = {ref.model_dump_json(): ref for ref in result.refs}
        for key in sorted(constraint_keys):
            refs.update({ref.model_dump_json(): ref for ref in values[key].refs})
        return MoneyValue(result.amount, result.currency, sources, tuple(refs.values()))


def compute_amount_units(case, decisions):
    programs = {p.object_id: p for p in case.amount_programs}
    units, issues, traces = [], [], []
    for unit in case.amount_units:
        program = programs.get(unit.object_id)
        if program is None or decisions[unit.object_id] == "reject":
            units.append(unit)
            continue
        try:
            if program.compile_status != "verified":
                raise ValueError("amount policy applicability requires review")
            basis = BoundMoneyExpression(case, program.basis).evaluate()
            limit = (
                BoundMoneyExpression(case, program.limit).evaluate()
                if program.limit is not None
                else None
            )
            if basis.currency != unit.currency or (
                limit is not None and limit.currency != unit.currency
            ):
                raise ValueError(
                    "calculated amount currency differs from the claim unit"
                )
            quantum = Decimal(program.quantum)
            with localcontext() as context:
                context.prec = 50
                basis_amount = basis.amount.quantize(quantum, rounding=ROUND_HALF_UP)
                limit_amount = (
                    limit.amount.quantize(quantum, rounding=ROUND_HALF_UP)
                    if limit is not None
                    else None
                )
            units.append(
                unit.model_copy(
                    update={
                        "material_supported_max": format(basis_amount, "f"),
                        "limit_status": "known" if limit is not None else "unlimited",
                        "limit": format(limit_amount, "f")
                        if limit is not None
                        else None,
                    }
                )
            )
            traces.append(
                {
                    "object_id": unit.object_id,
                    "material_supported_max": format(basis_amount, "f"),
                    "raw_material_supported_max": format(basis.amount, "f"),
                    "quantum": program.quantum,
                    "limit": format(limit_amount, "f") if limit is not None else None,
                    "source_ids": sorted(
                        basis.sources
                        | (limit.sources if limit is not None else frozenset())
                    ),
                    "refs": [
                        r.model_dump()
                        for r in basis.refs + (limit.refs if limit is not None else ())
                    ],
                }
            )
        except ValueError as error:
            units.append(
                unit.model_copy(
                    update={
                        "material_supported_max": None,
                        "limit_status": "unknown",
                        "limit": None,
                    }
                )
            )
            issues.append(
                {
                    "object_id": unit.object_id,
                    "reason": "amount_program_unresolved",
                    "detail": str(error),
                }
            )
    return units, issues, traces
