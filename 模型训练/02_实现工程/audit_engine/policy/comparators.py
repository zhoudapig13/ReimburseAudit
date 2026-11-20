"""Execute only declared policy comparisons on complete, typed operands."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation

OPERATIONS = frozenset({"EQ", "LE", "GE", "WITHIN", "IN_SET", "NOT_IN_SET", "AUTHORIZED", "CLOSED_ROUTE"})
Value = str | tuple[str, ...] | None


def normalize_text(value: str, mode: str = "exact") -> str:
    if mode == "casefold_alnum":
        return "".join(c for c in value.casefold() if c.isalnum())
    if mode == "casefold_whitespace":
        return " ".join(value.casefold().split())
    if mode == "exact":
        return value
    raise ValueError("unknown text normalization")


@dataclass(frozen=True)
class Comparison:
    complete: bool
    passed: bool | None
    reason: str | None = None


def _decimal(value: Value) -> Decimal:
    if not isinstance(value, str) or not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value):
        raise ValueError("operand must be an exact finite decimal string")
    result = Decimal(value)
    if not result.is_finite():
        raise ValueError("operand is not finite")
    return result


def _date(value: Value) -> date:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("date must already be unambiguous ISO YYYY-MM-DD")
    return date.fromisoformat(value)


def _datetime(value: Value) -> datetime:
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})?", value
    ):
        raise ValueError("timestamp must be an unambiguous ISO datetime")
    return datetime.fromisoformat(value)


def evaluate(
    operation: str,
    left: Value,
    right: Value,
    *,
    value_type: str | None = None,
    left_currency: str | None = None,
    right_currency: str | None = None,
    inclusive: bool = True,
    grace_days: int = 0,
    text_normalization: str = "exact",
) -> Comparison:
    """Unknown or invalid operands stay unresolved; never guess units or dates.

    AUTHORIZED is exact membership of an approver in an explicitly supplied
    authorized set. Hierarchies, amount-dependent authority and policy scope
    require the compiler to resolve a verified set before calling this function.
    """
    if operation not in OPERATIONS:
        return Comparison(False, None, "operation_outside_whitelist")
    if text_normalization not in {"exact", "casefold_whitespace", "casefold_alnum"}:
        return Comparison(False, None, "unknown_text_normalization")
    if left is None or right is None:
        return Comparison(False, None, "operand_not_obtained")
    if value_type is None:
        value_type = "decimal" if operation in {"LE", "GE"} else "text"
        if operation == "WITHIN":
            value_type = "date"
    try:
        def text(value):
            return normalize_text(value, text_normalization)

        if value_type == "money":
            if not all(
                isinstance(c, str) and re.fullmatch(r"[A-Z]{3}", c)
                for c in (left_currency, right_currency)
            ):
                return Comparison(False, None, "currency_not_confirmed")
            if left_currency != right_currency:
                return Comparison(
                    False, None, "currency_mismatch_no_verified_conversion"
                )
        if operation == "CLOSED_ROUTE":
            if value_type != "text" or not isinstance(left, tuple) or not left or not isinstance(right, str) or not right.strip():
                return Comparison(False, None, "source_routes_and_project_base_required")
            edges = []
            for route in left:
                if not isinstance(route, str):
                    return Comparison(False, None, "invalid_source_route")
                cities = route.replace("→", "->").split("->")
                if len(cities) != 2 or not all(city.strip() for city in cities):
                    return Comparison(False, None, "invalid_source_route")
                edges.append(tuple(text(city.strip()) for city in cities))
            incoming, outgoing, neighbors = Counter(), Counter(), defaultdict(set)
            for departure, arrival in edges:
                outgoing[departure] += 1
                incoming[arrival] += 1
                neighbors[departure].add(arrival)
                neighbors[arrival].add(departure)
            base = text(right.strip())
            reached, queue = set(), [base]
            while queue:
                city = queue.pop()
                if city not in reached:
                    reached.add(city)
                    queue.extend(neighbors[city] - reached)
            vertices = set(incoming) | set(outgoing)
            passed = base in vertices and reached == vertices and all(incoming[v] == outgoing[v] for v in vertices)
        elif operation in {"LE", "GE"}:
            if value_type not in {"decimal", "money", "date", "datetime"}:
                return Comparison(False, None, "incompatible_operand_type")
            convert = {"date": _date, "datetime": _datetime}.get(value_type, _decimal)
            lhs, rhs = convert(left), convert(right)
            passed = lhs <= rhs if operation == "LE" else lhs >= rhs
        elif operation == "EQ":
            if value_type in {"decimal", "money"}:
                passed = _decimal(left) == _decimal(right)
            elif value_type == "date":
                passed = _date(left) == _date(right)
            elif value_type == "datetime":
                lhs, rhs = _datetime(left), _datetime(right)
                if (lhs.tzinfo is None) != (rhs.tzinfo is None):
                    return Comparison(
                        False, None, "timestamp_timezone_context_mismatch"
                    )
                passed = lhs == rhs
            elif (
                value_type == "text"
                and isinstance(left, str)
                and isinstance(right, str)
            ):
                passed = text(left) == text(right)
            else:
                return Comparison(False, None, "incompatible_operand_type")
        elif operation == "WITHIN":
            if (
                value_type not in {"date", "datetime"}
                or not isinstance(right, tuple)
                or len(right) != 2
            ):
                return Comparison(False, None, "verified_date_interval_required")
            if (
                not isinstance(inclusive, bool)
                or type(grace_days) is not int
                or grace_days < 0
            ):
                return Comparison(False, None, "invalid_interval_policy")
            convert = _datetime if value_type == "datetime" else _date
            item, start, end = convert(left), convert(right[0]), convert(right[1])
            if (
                value_type == "datetime"
                and len({v.tzinfo is None for v in (item, start, end)}) != 1
            ):
                return Comparison(False, None, "timestamp_timezone_context_mismatch")
            if start > end:
                return Comparison(False, None, "reversed_date_interval")
            start, end = (
                start - timedelta(days=grace_days),
                end + timedelta(days=grace_days),
            )
            passed = start <= item <= end if inclusive else start < item < end
        else:
            if (
                value_type != "text"
                or not isinstance(left, str)
                or not isinstance(right, tuple)
                or not all(isinstance(v, str) for v in right)
            ):
                return Comparison(False, None, "verified_text_set_required")
            passed = text(left) in {text(item) for item in right}
            if operation == "NOT_IN_SET":
                passed = not passed
    except (ValueError, TypeError, InvalidOperation, OverflowError):
        return Comparison(False, None, "invalid_or_ambiguous_operand")
    return Comparison(True, bool(passed))
