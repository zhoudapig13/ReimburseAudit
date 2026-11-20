"""Append-only records of operations that actually ran."""

from __future__ import annotations

from datetime import datetime, timezone


class AuditTrace:
    def __init__(self):
        self.events: list[dict] = []

    def add(
        self,
        operation: str,
        inputs: list[str],
        outputs: list[str],
        *,
        condition_id: str | None = None,
        round_index: int = 0,
        details: dict | None = None,
    ):
        self.events.append(
            {
                "s": len(self.events) + 1,
                "op": operation,
                "in": inputs,
                "out": outputs,
                "time": datetime.now(timezone.utc).isoformat(),
                "condition_id": condition_id,
                "round": round_index,
                "details": details or {},
            }
        )
