"""Balance decision and amount supervision against long explanation strings."""

from __future__ import annotations

import re

import torch


def decision_amount_weights(tokenizer, answer, response_ids):
    """Allocate 25% loss to FLAG, 25% to amount, 50% to remaining tokens."""
    encoded = tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    if response_ids[: len(ids)].tolist() != ids:
        raise ValueError("answer token offsets do not match the supervised response")
    flag = re.search(r'"FLAG":([01])(?=,)', answer)
    amount = re.search(r'"max_legal":"([^"\\]+)"', answer)
    if not flag or not amount:
        raise ValueError("official answer decision or amount span missing")
    groups = []
    for match in (flag, amount):
        start, end = match.span(1)
        indices = {
            index
            for index, (left, right) in enumerate(encoded["offset_mapping"])
            if left < end and right > start
        }
        if not indices:
            raise ValueError("decision/amount has no supervised tokens")
        groups.append(indices)
    if groups[0] & groups[1]:
        raise ValueError("decision and amount token spans overlap")
    groups.append(set(range(len(response_ids))) - groups[0] - groups[1])
    if not groups[2]:
        raise ValueError("remaining answer tokens missing")
    weights = torch.zeros(len(response_ids), dtype=torch.float32)
    for group, mass in zip(groups, (0.25, 0.25, 0.5)):
        weights[torch.tensor(sorted(group), dtype=torch.long)] = mass / len(group)
    return weights
