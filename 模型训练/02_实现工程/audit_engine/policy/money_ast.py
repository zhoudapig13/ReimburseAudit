"""Bounded money-expression interpreter; values are supplied by cited readers."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, localcontext


def evaluate_money_ast(node: dict, values: dict[str, str]) -> Decimal:
    budget = [128]

    def number(value):
        try:
            value = Decimal(value)
        except (InvalidOperation, TypeError, ValueError):
            raise ValueError("invalid monetary operand") from None
        if (
            not value.is_finite()
            or value < 0
            or value.adjusted() > 18
            or value.as_tuple().exponent < -12
        ):
            raise ValueError("monetary operand out of range")
        return value

    def visit(expr, depth=0):
        budget[0] -= 1
        if depth > 16 or budget[0] < 0 or not isinstance(expr, dict):
            raise ValueError("money expression exceeds structural budget")
        op = expr.get("op")
        if op == "VALUE":
            if (
                not set(expr).issubset({"op", "source", "quantity_source"})
                or "source" not in expr
            ):
                raise ValueError("VALUE requires a cited source identifier")
            if expr["source"] not in values:
                raise ValueError("missing source value")
            result = number(values[expr["source"]])
            if "quantity_source" in expr:
                if expr["quantity_source"] not in values:
                    raise ValueError("missing explicit quantity")
                result *= number(values[expr["quantity_source"]])
            return number(result)
        if set(expr) != {"op", "args"} or not isinstance(expr["args"], list):
            raise ValueError("invalid money expression shape")
        args = [visit(child, depth + 1) for child in expr["args"]]
        if op in {"SUM", "MIN"} and args:
            return number(sum(args, Decimal(0)) if op == "SUM" else min(args))
        if op == "CAP" and len(args) == 2:
            return min(args)
        if op == "EXCLUDE" and len(args) == 2:
            return max(Decimal(0), args[0] - args[1])
        if op == "CONVERT" and len(args) == 2:
            return number(args[0] * args[1])
        raise ValueError("money operation or arity is outside the whitelist")

    with localcontext() as context:
        context.prec = 50
        return visit(node)
