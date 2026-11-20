"""Reversible spreadsheet cell serialization, without selecting policy rows."""

from __future__ import annotations

import hashlib
import json
import re

FORMAT = "xlsx_dense_cells_v1"
DESCRIPTION = (
    "Complete spreadsheet cells, no rows filtered. Each sheet has name n and rows r. "
    "Each row is [source_segment_id, Excel_row_number, values, optional_metadata]. "
    "Values are ordered by Excel columns A,B,C,...; null denotes an absent cell. "
    "Metadata entries [zero_based_column, Excel_value_type, format_index] override "
    "the defaults (JSON string=s, number=n, boolean=b; number format=General). "
    "Formats are literal Excel number formats, not executable instructions."
)


def column_index(coordinate: str, row: int) -> int:
    match = re.fullmatch(r"([A-Z]+)([1-9][0-9]*)", coordinate)
    if not match or int(match[2]) != row:
        raise ValueError("cell coordinate does not match its source row")
    index = 0
    for letter in match[1]:
        index = index * 26 + ord(letter) - ord("A") + 1
    if index > 16384:
        raise ValueError("column exceeds Excel limits")
    return index - 1


def column_name(index: int) -> str:
    value, name = index + 1, ""
    while value:
        value, digit = divmod(value - 1, 26)
        name = chr(ord("A") + digit) + name
    return name


def default_type(value) -> str:
    if isinstance(value, str):
        return "s"
    if isinstance(value, bool):
        return "b"
    if isinstance(value, (int, float)):
        return "n"
    raise ValueError("unsupported or null cell value")


def cell_projection(evidence: dict) -> list[dict]:
    if evidence["type"] != "xlsx":
        raise ValueError("not a spreadsheet")
    projection = []
    for segment in evidence["segments"]:
        if segment["kind"] != "xlsx_row":
            raise ValueError("unsupported spreadsheet segment")
        projection.append(
            {
                "segment_id": segment["segment_id"],
                "source_locator": segment["source_locator"],
                "cells": segment["cells"],
            }
        )
    return projection


def digest_projection(projection: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps(
            projection,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def pack_table(evidence: dict) -> dict:
    original = cell_projection(evidence)
    formats, sheets = ["General"], []
    seen_sheets, seen_segments = set(), set()
    for segment in original:
        locator = segment["source_locator"]
        if set(locator) != {"sheet", "row"}:
            raise ValueError("unsupported source locator metadata")
        name, row = locator["sheet"], locator["row"]
        if not sheets or sheets[-1]["n"] != name:
            if name in seen_sheets:
                raise ValueError("noncontiguous sheet segments")
            sheets.append({"n": name, "r": []})
            seen_sheets.add(name)
        if segment["segment_id"] in seen_segments:
            raise ValueError("duplicate source segment")
        seen_segments.add(segment["segment_id"])
        values, metadata, indices = [], [], []
        for cell in segment["cells"]:
            if set(cell) != {"coordinate", "value", "value_type", "number_format"}:
                raise ValueError("unsupported source cell metadata")
            index = column_index(cell["coordinate"], row)
            if indices and index <= indices[-1]:
                raise ValueError("source cells are not uniquely ordered")
            indices.append(index)
            values.extend([None] * (index + 1 - len(values)))
            values[index] = cell["value"]
            inferred = default_type(cell["value"])
            if cell["number_format"] not in formats:
                formats.append(cell["number_format"])
            format_index = formats.index(cell["number_format"])
            if cell["value_type"] != inferred or format_index:
                metadata.append([index, cell["value_type"], format_index])
        record = [segment["segment_id"], row, values]
        if metadata:
            record.append(metadata)
        sheets[-1]["r"].append(record)
    packed = {
        "format": FORMAT,
        "formats": formats,
        "sheets": sheets,
        "source_cells_sha256": digest_projection(original),
    }
    if unpack_table(packed) != original:
        raise ValueError("spreadsheet cell roundtrip mismatch")
    return packed


def unpack_table(packed: dict) -> list[dict]:
    if packed["format"] != FORMAT:
        raise ValueError("unsupported compact table format")
    result = []
    for sheet in packed["sheets"]:
        for record in sheet["r"]:
            segment_id, row, values = record[:3]
            overrides = (
                {entry[0]: entry[1:] for entry in record[3]} if len(record) == 4 else {}
            )
            cells = []
            for index, value in enumerate(values):
                if value is None:
                    continue
                kind, format_index = overrides.get(index, [default_type(value), 0])
                cells.append(
                    {
                        "coordinate": f"{column_name(index)}{row}",
                        "value": value,
                        "value_type": kind,
                        "number_format": packed["formats"][format_index],
                    }
                )
            result.append(
                {
                    "segment_id": segment_id,
                    "source_locator": {"sheet": sheet["n"], "row": row},
                    "cells": cells,
                }
            )
    if digest_projection(result) != packed["source_cells_sha256"]:
        raise ValueError("compact spreadsheet does not match its source digest")
    return result


def table_text(evidence: dict) -> str:
    return (
        DESCRIPTION
        + "\n"
        + json.dumps(
            pack_table(evidence),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
