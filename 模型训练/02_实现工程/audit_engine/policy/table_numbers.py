"""Aggregate automatically parsed spreadsheet rows with explicit source keys."""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation

from audit_engine.contracts import Citation
from audit_engine.readers.value_formats import date_value


def header_key(value):
    return "".join(value.casefold().split())


def source_rows(material, sheet):
    for segment in material.segments:
        match = re.fullmatch(r"Sheet ([^:]+): (.*)", segment.text, flags=re.DOTALL)
        if not match or match.group(1) != sheet:
            continue
        cells = {}
        for piece in re.split(r"\s+\|\s+(?=[A-Z]+\d+=)", match.group(2)):
            cell = re.fullmatch(r"([A-Z]+)(\d+)=(.*)", piece, flags=re.DOTALL)
            if not cell:
                raise ValueError("spreadsheet segment does not preserve cell identities")
            cells[cell.group(1)] = cell.group(3)
        yield segment, cells


def sum_source_table(material, binding, period=None):
    rows = list(source_rows(material, binding.sheet))
    required = [binding.column, *binding.row_keys, *binding.required_nonempty, *binding.required_contains]
    if binding.date_key:
        required.append(binding.date_key)
    if binding.group_key:
        required.append(binding.group_key)
    headers = []
    for index, (segment, cells) in enumerate(rows):
        mapping = {}
        for column, value in cells.items():
            key = header_key(value)
            if key in mapping:
                raise ValueError("spreadsheet header labels are ambiguous")
            mapping[key] = column
        if all(header_key(name) in mapping for name in required):
            headers.append((index, segment, mapping))
    if len(headers) != 1:
        raise ValueError("source table requires one unambiguous matching header")
    header_index, header, mapping = headers[0]
    column = mapping[header_key(binding.column)]
    keys = [mapping[header_key(name)] for name in binding.row_keys]
    day_column = mapping[header_key(binding.date_key)] if binding.date_key else None
    group_column = mapping[header_key(binding.group_key)] if binding.group_key else None
    signatures = [mapping[header_key(name)] for name in binding.required_nonempty]
    seen, selected, grouped = set(), [], {}
    total = Decimal(0)
    identified = 0
    allowed = {Decimal(v) for v in binding.allowed_values}
    for segment, cells in rows[header_index + 1:]:
        item_keys = tuple(cells.get(key, "").strip() for key in keys)
        if not all(item_keys):
            # Fully identified records cannot disappear silently. A footer lacks
            # the required identity/date keys and is not a data record.
            if column in cells and cells[column].strip() and any(item_keys):
                if day_column and date_value(cells.get(day_column, "")) is None:
                    continue
                raise ValueError("incomplete source table record keys")
            continue
        if day_column:
            iso = date_value(cells.get(day_column, ""))
            if iso is None:
                raise ValueError("source record has an ambiguous date")
            day = date.fromisoformat(iso)
            identified += 1
            if period is not None and not period[0] <= day <= period[1]:
                continue
        else:
            identified += 1
        if item_keys in seen:
            raise ValueError("duplicate source table record keys")
        seen.add(item_keys)
        if any(not cells.get(key, "").strip() for key in signatures):
            raise ValueError("required source table signature/acknowledgement is absent")
        if any(marker.casefold() not in cells.get(mapping[header_key(name)], "").casefold()
               for name, marker in binding.required_contains.items()):
            raise ValueError("source table acknowledgement does not contain its required marker")
        raw = cells.get(column, "").strip().replace(",", "")
        try:
            value = Decimal(raw)
        except (InvalidOperation, ValueError):
            raise ValueError("source table quantity is missing or nonnumeric") from None
        if not value.is_finite() or value < 0 or value.adjusted() > 18 or value.as_tuple().exponent < -12:
            raise ValueError("source table quantity is outside the supported range")
        if allowed and value not in allowed:
            raise ValueError("source table quantity is outside its declared coding")
        total += value
        if group_column:
            group = cells[group_column].strip()
            grouped[group] = grouped.get(group, Decimal(0)) + value
        selected.append(segment)
    # An empty but structurally valid period has zero eligible records. The
    # interval and full source remain dependencies of that result.
    if identified == 0:
        raise ValueError("source table has no identified data records")
    refs = [Citation(evidence_id=material.evidence_id, segment_id=header.segment_id,
                     quote=header.text, note=f"Deterministic sum of {len(selected)} identified records from sheet {binding.sheet}.")]
    for segment in ([selected[0], selected[-1]] if selected else []):
        refs.append(Citation(evidence_id=material.evidence_id, segment_id=segment.segment_id, quote=segment.text))
    return total, grouped, tuple(refs)
