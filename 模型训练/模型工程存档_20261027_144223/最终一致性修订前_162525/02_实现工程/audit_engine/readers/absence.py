"""Bind explicit whole-image date-absence observations to declared date roles."""

import hashlib
import json
import re
from pathlib import Path

from PIL import Image, ImageOps

from audit_engine.contracts import AuditCase, Operand
from audit_engine.readers.value_formats import explicit_printed_date_absence


def bind_explicit_date_absence(case: AuditCase, source_root: Path):
    if not case.input_version.startswith("patent_reader_v1:"):
        return case, []
    raw, records = case.model_dump(), []
    for material in case.evidence:
        if material.type != "image" or material.read_status != "obtained":
            continue
        refs = {ref.model_dump_json(): ref for observation in case.operands
                if observation.evidence_id == material.evidence_id
                for ref in observation.refs
                if ref.kind == "visual" and explicit_printed_date_absence(ref.note)}
        if not refs:
            continue
        source = (source_root / material.raw_ref).resolve()
        if not source.is_relative_to(source_root.resolve()) or hashlib.sha256(source.read_bytes()).hexdigest() != material.source_sha256:
            raise ValueError("date-absence source changed or is outside the source root")
        with Image.open(source) as image:
            picture = ImageOps.exif_transpose(image)
            width, height = picture.size
        # Long receipts are tiled by image_views. Absence in a tile does not
        # establish absence from the original whole receipt.
        if height > 3 * width:
            continue
        for condition in case.conditions:
            for role in condition.roles:
                definition = condition.role_definitions.get(role, "").casefold()
                binding = re.search(r"(?:^|_)(e\d+)(?:_|$)", role)
                date_role = condition.role_types.get(role) == "date" or (
                    condition.operation and condition.operation.value_type == "date"
                    and condition.operation.left_role == role)
                if ("date" not in definition or "print" not in definition
                        or not date_role
                        or role in condition.derived_roles
                        or condition.role_origins.get(role) == "background"
                        or not binding or binding.group(1) != material.evidence_id):
                    continue
                existing = [o for o in case.operands if o.condition_id == condition.condition_id
                            and o.role == role and o.evidence_id == material.evidence_id]
                if any(o.status == "obtained" for o in existing):
                    continue  # Competing reads need a reread, never an override.
                if not any(o.value_type == "date" and o.status == "not_obtained"
                           and any(ref.kind == "visual" and explicit_printed_date_absence(ref.note)
                                   for ref in o.refs) for o in existing):
                    observation = Operand.model_validate({"condition_id": condition.condition_id,
                        "role": role, "evidence_id": material.evidence_id, "value": None,
                        "value_type": "date", "status": "not_obtained", "depends_on": [material.evidence_id],
                        "refs": [ref.model_dump() for ref in refs.values()]})
                    raw["operands"].append(observation.model_dump())
                marker = f"confirmed_missing:{condition.condition_id}:{role}"
                if marker not in raw["unresolved"]:
                    raw["unresolved"].append(marker)
                records.append({"evidence_id": material.evidence_id, "condition_id": condition.condition_id,
                    "role": role, "source_sha256": material.source_sha256, "whole_image": True,
                    "source_observations": [ref.model_dump() for ref in refs.values()],
                    "method": "bind_explicit_whole_image_printed_date_absence_v1",
                    "reference_answers_read": False})
    if records:
        raw["unresolved"] = sorted(set(raw["unresolved"]))
        raw["input_version"] = "patent_reader_v1:bound_printed_date_absence:" + hashlib.sha256(
            json.dumps(records, sort_keys=True).encode()).hexdigest()
    return AuditCase.model_validate(raw), records
