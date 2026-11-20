"""Reversible columnar JSON records, retaining order and repeated entries."""

from __future__ import annotations

import hashlib
import json

FORMAT = "json_records_columns_v1"
DESCRIPTION = (
    "Complete original JSON records in columnar form, without filtering or deduplication. "
    "columns lists each original field name. For one column, each rows element is the "
    "original field value; otherwise each row is a list of values in column order. "
    "Record order, repeated records, value types and leading zeros are preserved."
)


def digest_value(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def pack_records(value: list[dict]) -> dict:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(row, dict) for row in value)
    ):
        raise ValueError("columnar encoding requires a nonempty record list")
    columns = list(value[0])
    if not columns or any(set(row) != set(columns) for row in value):
        raise ValueError("record fields are not uniform")
    rows = (
        [row[columns[0]] for row in value]
        if len(columns) == 1
        else [[row[key] for key in columns] for row in value]
    )
    packed = {
        "format": FORMAT,
        "columns": columns,
        "rows": rows,
        "source_value_sha256": digest_value(value),
    }
    if unpack_records(packed) != value:
        raise ValueError("JSON records roundtrip mismatch")
    return packed


def unpack_records(packed: dict) -> list[dict]:
    if (
        packed["format"] != FORMAT
        or not packed["columns"]
        or len(set(packed["columns"])) != len(packed["columns"])
    ):
        raise ValueError("invalid columnar JSON schema")
    columns = packed["columns"]
    if len(columns) == 1:
        result = [{columns[0]: value} for value in packed["rows"]]
    else:
        if any(
            not isinstance(row, list) or len(row) != len(columns)
            for row in packed["rows"]
        ):
            raise ValueError("columnar row width mismatch")
        result = [dict(zip(columns, row, strict=True)) for row in packed["rows"]]
    if digest_value(result) != packed["source_value_sha256"]:
        raise ValueError("columnar JSON does not match its source digest")
    return result


def records_text(evidence: dict) -> str:
    if evidence["type"] != "json":
        raise ValueError("not a JSON material")
    return (
        DESCRIPTION
        + "\n"
        + json.dumps(
            pack_records(json.loads(evidence["text"])),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
