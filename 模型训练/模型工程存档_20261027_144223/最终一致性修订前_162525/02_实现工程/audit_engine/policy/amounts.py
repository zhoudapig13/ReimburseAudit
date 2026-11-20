"""Exact Decimal projection of disjoint amount units into A_ok/A_no/A_review."""

from __future__ import annotations

from decimal import Decimal

from audit_engine.contracts import AmountUnit


def project_amounts(
    units: list[AmountUnit],
    states: dict[str, str],
    claimed_amount: str,
    currency: str,
) -> dict:
    """The compiler must collapse shared caps into a single group unit first.

    states maps each amount unit to accept/reject/review. Unknown caps always
    force review. Maximum supported amount is recorded separately from the
    partition of the submitted claim, so it may exceed the submitted claim.
    """
    claim = Decimal(claimed_amount)
    if len({u.object_id for u in units}) != len(units):
        raise ValueError("duplicate amount unit")
    if set(states) != {u.object_id for u in units}:
        raise ValueError("amount status coverage differs from the unit set")
    if any(u.currency != currency for u in units):
        raise ValueError("amount units require the same confirmed currency")
    if sum((Decimal(u.claimed_amount) for u in units), Decimal(0)) != claim:
        raise ValueError("amount units do not exactly partition the claim")
    groups = [u.shared_group_id for u in units if u.shared_group_id]
    if len(groups) != len(set(groups)):
        raise ValueError("combine shared-cap members into one group before projection")
    totals = {"A_ok": Decimal(0), "A_no": Decimal(0), "A_review": Decimal(0)}
    supported = Decimal(0)
    maximum_known = True
    details = []
    for unit in units:
        amount = Decimal(unit.claimed_amount)
        limit = Decimal(unit.limit) if unit.limit is not None else None
        usable = min(amount, limit) if limit is not None else amount
        state = states[unit.object_id]
        if state not in {"accept", "reject", "review"}:
            raise ValueError("invalid unit decision")
        if unit.limit_status == "unknown" and state != "reject":
            state = "review"
        if state == "accept" and unit.material_supported_max is None:
            state = "review"
        if unit.material_supported_max is not None:
            usable = min(usable, Decimal(unit.material_supported_max))
        key = {"accept": "A_ok", "reject": "A_no", "review": "A_review"}[state]
        totals[key] += usable
        totals["A_no"] += amount - usable
        if state == "reject":
            unit_maximum = Decimal(0)
        elif state == "review" or unit.material_supported_max is None:
            maximum_known = False
            unit_maximum = None
        else:
            unit_maximum = Decimal(unit.material_supported_max)
            if limit is not None:
                unit_maximum = min(unit_maximum, limit)
            supported += unit_maximum
        details.append(
            {
                "object_id": unit.object_id,
                "state": state,
                "claim": str(amount),
                "within_limit": str(usable),
                "material_supported_max": None
                if unit_maximum is None
                else str(unit_maximum),
            }
        )
    assert sum(totals.values()) == claim
    return {
        "currency": currency,
        "A_claim": str(claim),
        **{k: str(v) for k, v in totals.items()},
        "A_max": str(supported) if maximum_known else None,
        "units": details,
        "conservation_valid": True,
    }
