"""Validate pending policy proposals without promoting them to supervision."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

from audit_engine.readers.quotes import normalize_whitespace, verify_text_quote

FIELDS = {
    "question",
    "kind",
    "roles",
    "operation",
    "policy_quote",
    "policy_refs",
    "scope",
    "amount_unit",
    "exceptions",
    "consequence",
}
OPERATIONS = {"EQ", "LE", "GE", "WITHIN", "IN_SET", "AUTHORIZED"}


def validate_policy_output(raw: str) -> dict:
    proposal = json.loads(raw)
    if not isinstance(proposal, dict) or set(proposal) != {
        "conditions",
        "uncertainties",
    }:
        raise ValueError("unexpected policy proposal fields")
    if not isinstance(proposal["conditions"], list) or not isinstance(
        proposal["uncertainties"], list
    ):
        raise ValueError("conditions and uncertainties must be lists")
    if not all(isinstance(value, str) for value in proposal["uncertainties"]):
        raise ValueError("uncertainties must contain strings")
    for condition in proposal["conditions"]:
        if not isinstance(condition, dict) or set(condition) != FIELDS:
            raise ValueError("unexpected condition fields")
        if (
            not isinstance(condition["question"], str)
            or not condition["question"].strip()
        ):
            raise ValueError("empty compliance question")
        if condition["kind"] not in {"required", "prohibited", "amount", "open_text"}:
            raise ValueError("invalid condition kind")
        roles = condition["roles"]
        if (
            not isinstance(roles, list)
            or not 1 <= len(roles) <= 8
            or not all(isinstance(r, str) and r.strip() for r in roles)
        ):
            raise ValueError("condition requires one to eight named roles")
        if (
            condition["operation"] is not None
            and condition["operation"] not in OPERATIONS
        ):
            raise ValueError("operation outside whitelist")
        if (
            not isinstance(condition["policy_quote"], str)
            or not condition["policy_quote"].strip()
        ):
            raise ValueError("policy condition requires a literal quote")
        refs = condition["policy_refs"]
        if not isinstance(refs, list) or not refs:
            raise ValueError("policy condition requires segment references")
        for ref in refs:
            if (
                not isinstance(ref, dict)
                or set(ref) != {"segment_id", "quote"}
                or not all(isinstance(ref[k], str) and ref[k].strip() for k in ref)
            ):
                raise ValueError("invalid policy segment reference")
        for name in ("scope", "amount_unit", "exceptions", "consequence"):
            if condition[name] is not None and not isinstance(condition[name], str):
                raise ValueError(f"invalid {name} value")
    return proposal


def annotate_source_view(text: str, material: dict) -> str:
    """Insert IDs around complete unique segments without removing view text."""
    segments = material.get("segments", [])
    counts = Counter(s["text"] for s in segments)
    spans = []
    for segment in segments:
        value = segment["text"]
        if value and counts[value] == 1 and text.count(value) == 1:
            start = text.index(value)
            spans.append((start, start + len(value), segment["segment_id"]))
    pieces, cursor = [], 0
    for start, end, identifier in sorted(spans):
        if start < cursor:
            continue
        pieces.extend(
            [
                text[cursor:start],
                f"[segment {identifier}]",
                text[start:end],
                "[/segment]",
            ]
        )
        cursor = end
    return "".join(pieces) + text[cursor:]


def split_existing_quote(quote: str, material: dict) -> list[dict]:
    """Split an exact unique quote already present in the parser's full text.

    Never concatenate segments to find a quote, change values, or fuzzy match.
    Every non-whitespace character must map to a separately verified segment.
    """
    full_text = material.get("text")
    if not isinstance(full_text, str):
        return []
    source, needle = normalize_whitespace(full_text), normalize_whitespace(quote)
    if not needle or source.count(needle) != 1:
        return []
    start = source.index(needle)
    end = start + len(needle)
    covered = [False] * len(needle)
    references, cursor = [], 0
    for segment in material.get("segments", []):
        text = normalize_whitespace(segment["text"])
        position = source.find(text, cursor) if text else -1
        if position < 0:
            return []
        cursor = position + len(text)
        left, right = max(start, position), min(end, cursor)
        if left >= right:
            continue
        reference = {
            "evidence_id": material["evidence_id"],
            "segment_id": segment["segment_id"],
            "quote": source[left:right],
            "kind": "text",
        }
        check = verify_text_quote(reference, [material])
        if not check.existence_verified:
            return []
        references.append(
            {
                **reference,
                **check.to_dict(),
                "resolution": "split_existing_contiguous_quote",
            }
        )
        covered[left - start : right - start] = [True] * (right - left)
        if cursor >= end:
            break
    if not references or any(
        not ok and not ch.isspace() for ok, ch in zip(covered, needle, strict=True)
    ):
        return []
    return references


def attach_source_checks(proposal: dict, material: dict) -> list[dict]:
    candidates = []
    for condition in proposal["conditions"]:
        references = []
        for proposed_ref in condition["policy_refs"]:
            reference = {
                "evidence_id": material["evidence_id"],
                **proposed_ref,
                "kind": "text",
            }
            check = verify_text_quote(reference, [material])
            checked = {**reference, **check.to_dict()}
            if not check.existence_verified:
                # Preserve the failed teacher citation and explicitly record
                # any literal segmentation of its unchanged original quote.
                resolved = split_existing_quote(reference["quote"], material)
                references.extend(
                    [{**r, "original_reference": checked} for r in resolved]
                    if resolved
                    else [checked]
                )
            else:
                references.append(checked)
        identity = hashlib.sha256(
            json.dumps(condition, sort_keys=True).encode()
        ).hexdigest()[:16]
        candidates.append(
            {
                "condition_id": f"proposal_{identity}",
                **condition,
                "policy_refs": references,
                "annotation_status": "pending",
                "approved_training_label": False,
                "compile_status": "pending",
                "semantic_verified": False,
                "issues": ["policy_semantics_and_applicability_need_review"]
                + (
                    ["policy_quote_not_found_in_one_source_segment"]
                    if not all(r["existence_verified"] for r in references)
                    else []
                ),
            }
        )
    return candidates
