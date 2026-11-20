"""Timing, data cursors and validation decisions for sustained official SFT."""

from __future__ import annotations

import math


def timed_learning_rate(step, elapsed_seconds, schedule_seconds, scale=1.0):
    """Warm up ten updates, decay over eight training hours, then hold a floor."""
    warmup = min(1.0, (step + 1) / 10)
    progress = min(1.0, max(0.0, elapsed_seconds / schedule_seconds))
    cosine = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    return warmup * cosine * scale


def next_cursor(epoch, order, next_offset):
    if next_offset >= len(order):
        return {"epoch_index": epoch + 1, "order": None, "next_offset": 0}
    return {"epoch_index": epoch, "order": list(order), "next_offset": next_offset}


def validation_rank(metrics):
    """Prefer jointly correct decisions/amounts, then field and format coverage."""
    return (
        metrics["joint_flag_amount_correct"],
        metrics["flag_correct"] + metrics["amount_correct"],
        metrics["schema_valid"],
        metrics["explanation_exact_match"],
    )


def plateau_decision(best, current, stale_rounds, reduced, patience=3):
    if best is None or validation_rank(current) > validation_rank(best):
        return {"action": "improved", "stale_rounds": 0}
    stale_rounds += 1
    if stale_rounds < patience:
        return {"action": "continue", "stale_rounds": stale_rounds}
    return {
        "action": "stop" if reduced else "reduce_learning_rate",
        "stale_rounds": stale_rounds if reduced else 0,
    }
