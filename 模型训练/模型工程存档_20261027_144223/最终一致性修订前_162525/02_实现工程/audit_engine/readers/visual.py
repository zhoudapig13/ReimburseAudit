"""Strict validation of provisional whole-image readings, never gold labels."""

from __future__ import annotations

import json


def validate_visual_output(raw: str) -> dict:
    value = json.loads(raw)
    fields = {
        "document_kind",
        "reading_status",
        "transcription",
        "facts",
        "visual_observations",
        "uncertainties",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("unexpected visual reading fields")
    if value["document_kind"] is not None and (
        not isinstance(value["document_kind"], str)
        or not value["document_kind"].strip()
    ):
        raise ValueError("document_kind must be a nonempty string or null")
    if value["reading_status"] not in {"clear", "partial", "needs_review"}:
        raise ValueError("invalid reading status")
    if value["transcription"] is not None and not isinstance(
        value["transcription"], str
    ):
        raise ValueError("transcription must be a string or null")
    if not isinstance(value["facts"], list):
        raise ValueError("facts must be a list")
    for fact in value["facts"]:
        if not isinstance(fact, dict) or set(fact) != {
            "field",
            "value",
            "currency",
            "source_text",
            "confidence",
        }:
            raise ValueError("unexpected fact fields")
        if not isinstance(fact["field"], str) or not fact["field"].strip():
            raise ValueError("fact field must be a nonempty string")
        if fact["value"] is not None and not isinstance(fact["value"], str):
            raise ValueError("raw value must be a string or null")
        if fact["confidence"] not in {"high", "medium", "low"}:
            raise ValueError("invalid confidence")
        if not all(
            fact[k] is None or isinstance(fact[k], str)
            for k in ("currency", "source_text")
        ):
            raise ValueError("currency and source_text must be strings or null")
    for field in ("visual_observations", "uncertainties"):
        if not isinstance(value[field], list) or not all(
            isinstance(s, str) and s.strip() for s in value[field]
        ):
            raise ValueError(f"{field} must be a list of nonempty strings")
    return value


def recover_completed_visual_fields(raw: str) -> dict | None:
    """Keep complete factual fields when only trailing observations were cut.

    No incomplete fact/amount is retained, no new source fact is generated, and
    the result remains explicitly partial with unavailable fields recorded.
    """
    text = raw.strip()
    if not text.startswith("{"):
        return None
    decoder, cursor, fields = json.JSONDecoder(), 1, {}
    allowed = {
        "document_kind",
        "reading_status",
        "transcription",
        "facts",
        "visual_observations",
        "uncertainties",
    }
    try:
        while True:
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor >= len(text):
                break
            if text[cursor] == "}":
                # A closed object or extra trailing text is not this failure.
                return None
            key, cursor = decoder.raw_decode(text, cursor)
            if not isinstance(key, str) or key not in allowed or key in fields:
                return None
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor >= len(text) or text[cursor] != ":":
                return None
            cursor += 1
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            try:
                value, cursor = decoder.raw_decode(text, cursor)
            except json.JSONDecodeError:
                if key not in {"visual_observations", "uncertainties"}:
                    return None
                break
            fields[key] = value
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor == len(text):
                break
            if text[cursor] != ",":
                return None
            cursor += 1
        if not {"document_kind", "reading_status", "transcription", "facts"} <= set(
            fields
        ):
            return None
        missing = sorted(allowed - set(fields))
        original_status = fields["reading_status"]
        fields.setdefault("visual_observations", [])
        fields.setdefault("uncertainties", [])
        # Validate all complete source values before adding processing metadata.
        validate_visual_output(json.dumps(fields))
        fields["reading_status"] = "partial"
        fields["uncertainties"] = [
            *fields["uncertainties"],
            "Generation was truncated after complete factual fields; trailing observations/uncertainties are unavailable and require review.",
        ]
        candidate = validate_visual_output(json.dumps(fields))
    except (ValueError, TypeError):
        return None
    return {
        "candidate": candidate,
        "unavailable_fields": missing,
        "original_reading_status": original_status,
        "source_output_complete": False,
        "recovery_method": "complete_top_level_fields_only",
    }
