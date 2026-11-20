"""Verify literal source quotes without turning teacher assertions into gold."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


@dataclass(frozen=True)
class QuoteCheck:
    evidence_id: str
    segment_id: str
    existence_verified: bool
    semantic_verified: bool
    status: str
    source_segment_sha256: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _has_literal_match(source: str, quote: str) -> bool:
    start = source.find(quote)
    while start >= 0:
        end = start + len(quote)
        left, right = source[:start], source[end:]
        cuts_word_left = quote[0].isalnum() and left and left[-1].isalnum()
        cuts_word_right = quote[-1].isalnum() and right and right[0].isalnum()
        cuts_number_left = (
            quote[0].isdigit()
            and len(left) >= 2
            and left[-1] in ".,/-"
            and left[-2].isdigit()
        )
        cuts_number_right = (
            quote[-1].isdigit()
            and len(right) >= 2
            and right[0] in ".,/-"
            and right[1].isdigit()
        )
        loses_sign = quote[0].isdigit() and left and left[-1] in "+-"
        if not (
            cuts_word_left
            or cuts_word_right
            or cuts_number_left
            or cuts_number_right
            or loses_sign
        ):
            return True
        start = source.find(quote, start + 1)
    return False


def verify_text_quote(
    reference: dict,
    evidence: list[dict],
    *,
    semantic_verified: bool = False,
) -> QuoteCheck:
    """Literal existence and semantic approval are independent checks.

    Only the explicit caller-side approval can mark semantic verification.
    Flags in a teacher/model reference are ignored. Image/OCR text existence
    does not verify the original visual facts and never approves a label.
    """
    if type(semantic_verified) is not bool:
        raise TypeError("semantic approval must be an explicit boolean")
    evidence_id, segment_id = reference.get("evidence_id"), reference.get("segment_id")
    if not isinstance(evidence_id, str) or not isinstance(segment_id, str):
        raise ValueError("text reference requires evidence and segment IDs")

    def check(status, digest=None):
        return QuoteCheck(evidence_id, segment_id, False, False, status, digest)

    if reference.get("kind", "text") != "text":
        return check("not_a_text_reference")
    matches = [
        material for material in evidence if material["evidence_id"] == evidence_id
    ]
    if len(matches) != 1:
        return check("source_missing_or_ambiguous")
    segments = [
        segment
        for segment in matches[0].get("segments", [])
        if segment["segment_id"] == segment_id
    ]
    if len(segments) != 1:
        return check("segment_missing_or_ambiguous")
    source = segments[0].get("text")
    quote = reference.get("quote")
    if not isinstance(source, str) or not isinstance(quote, str) or not quote.strip():
        return check("empty_or_invalid_quote")
    digest = hashlib.sha256(source.encode()).hexdigest()
    if not _has_literal_match(
        normalize_whitespace(source), normalize_whitespace(quote)
    ):
        return check("literal_quote_not_found", digest)
    return QuoteCheck(
        evidence_id,
        segment_id,
        True,
        semantic_verified,
        "verified" if semantic_verified else "semantic_review_pending",
        digest,
    )
