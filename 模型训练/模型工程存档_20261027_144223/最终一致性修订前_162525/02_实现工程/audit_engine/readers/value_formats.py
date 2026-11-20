"""Lossless normalization of unambiguous source dates; no locale guessing."""

import re
from datetime import date, datetime


def declared_categories(definition):
    """Read an explicit finite category enumeration, never infer new labels."""
    match = re.search(r"\bclassify\b[^.]*?\bas\s+([^.;]+)", definition or "", re.IGNORECASE)
    if not match:
        return ()
    labels = [value.strip() for value in re.split(r",\s*(?:or\s+)?|\s+or\s+", match.group(1))]
    if 2 <= len(labels) <= 8 and all(re.fullmatch(r"[a-z][a-z0-9_]*", label) for label in labels):
        return tuple(labels)
    return ()


def explicit_printed_date_absence(note):
    """Separate an explicit whole-source absence claim from an unreadable date."""
    if not isinstance(note, str) or re.search(
        r"unreadable|illegible|blurr|cropp|low.resolution|cannot (?:read|see)|not legible", note, re.IGNORECASE
    ):
        return False
    return bool(re.search(
        r"\bno\s+(?:(?:consumption|transaction|travel|calendar)\s+)?date\s+(?:(?:is|was)\s+)?(?:printed|shown|appears)\b"
        r"|\bdate\b[^.;]{0,24}\b(?:not (?:explicitly )?printed|is absent|is missing)\b",
        note, re.IGNORECASE))


def missing_required_field(note, definition):
    """Identify a completeness claim's explicit contradictory source qualifier."""
    for field in ("date", "address", "merchant name", "itemized expenses", "total"):
        if field not in definition.casefold():
            continue
        escaped = re.escape(field)
        if (re.search(rf"\b{escaped}\b[^.;]{{0,35}}\b(?:not (?:explicitly )?printed|missing|absent|not shown|not present|not visible)\b", note, re.IGNORECASE)
                or re.search(rf"\b(?:no|missing|absent)\s+(?:consumption\s+)?{escaped}\b", note, re.IGNORECASE)):
            return field
    return None


def date_value(value):
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        pass
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T00:00(?::00)?", value):
        return date_value(value[:10])
    numeric = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", value)
    if numeric:
        first, second, year = map(int, numeric.groups())
        if (first > 12) != (second > 12):
            day, month = (first, second) if first > 12 else (second, first)
            try:
                return date(year, month, day).isoformat()
            except ValueError:
                return None
    # Month names make ordering explicit. Ambiguous numeric dates and two-digit
    # years are deliberately excluded, regardless of the policy's date window.
    for fmt in (
        "%B %d, %Y",
        "%b %d, %Y",
        "%B %d %Y",
        "%b %d %Y",
        "%d %B %Y",
        "%d %b %Y",
    ):
        try:
            return datetime.strptime(value.strip(), fmt).date().isoformat()
        except ValueError:
            pass
    return None


def normalize_fact_value(value, value_type, status):
    if status != "obtained":
        return value, status
    if value_type == "date":
        normalized = date_value(value)
        return (normalized, status) if normalized else (value, "ambiguous")
    if value_type == "interval":
        parts = (
            value
            if isinstance(value, list)
            else re.split(r"\s+(?:to|[–—])\s+", value or "")
        )
        if len(parts) == 2:
            normalized = [date_value(part) for part in parts]
            if all(normalized):
                return normalized, status
        return value, "ambiguous"
    if value_type == "datetime":
        if isinstance(value, str) and re.fullmatch(
            r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})?", value
        ):
            try:
                return datetime.fromisoformat(value).isoformat(
                    timespec="seconds"
                ), status
            except ValueError:
                pass
        if isinstance(value, str):
            for fmt in (
                "%b %d,%Y@%I:%M%p",
                "%B %d,%Y@%I:%M%p",
                "%b %d, %Y@%I:%M%p",
                "%B %d, %Y@%I:%M%p",
            ):
                try:
                    return datetime.strptime(value, fmt).isoformat(
                        timespec="seconds"
                    ), status
                except ValueError:
                    pass
        return value, "ambiguous"
    return value, status
