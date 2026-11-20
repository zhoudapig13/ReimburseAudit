"""One source-grounded binding pass for record selection; no teacher inputs."""

from __future__ import annotations

from decimal import Decimal

from audit_engine.contracts import AuditCase, Operand
from audit_engine.readers.material_views import image_views
from audit_engine.readers.patent_reader import source_views


def value_key(observation):
    value = observation.value
    if observation.value_type in {"money", "decimal"} and isinstance(value, str):
        value = Decimal(value)
    elif isinstance(value, list):
        value = tuple(value)
    return observation.value_type, value, observation.currency


def check_reads(case: AuditCase, rows):
    accepted, issues = [], []
    for row in rows:
        observation = Operand.model_validate(row)
        try:
            case.model_copy(update={"operands": [observation]}).check_links()
        except ValueError as error:
            # Preserve the actual output in its call log; never replace an
            # incorrect value with a supervised or manually corrected value.
            observation = observation.model_copy(
                update={"status": "ambiguous", "value": None}
            )
            case.model_copy(update={"operands": [observation]}).check_links()
            issues.append(
                f"read_value_unverified:{observation.evidence_id}:{observation.condition_id}:{error}"
            )
        accepted.append(observation)
    return accepted, issues


def resolve_selection_reads(reader, case: AuditCase, config):
    """Resolve a missing/ambiguous role using primitive reads of other roles.

    Context only selects records; source files still supply every emitted value.
    The caller's frozen backbone encodings remain independent per material.
    """
    initial = list(case.operands)
    final = list(initial)
    issues = []
    for condition in case.conditions:
        if condition.compile_status != "verified":
            continue
        for role in condition.roles:
            current = [
                o
                for o in initial
                if o.condition_id == condition.condition_id and o.role == role
            ]
            observed = [o for o in current if o.status == "obtained"]
            if len({value_key(o) for o in observed}) == 1 and not any(
                o.status == "ambiguous" for o in current
            ):
                continue
            context = []
            for other in condition.roles:
                candidates = [
                    o
                    for o in initial
                    if o.condition_id == condition.condition_id
                    and o.role == other
                    and other != role
                    and o.status == "obtained"
                    and o.depends_on == [o.evidence_id]
                ]
                if len({value_key(o) for o in candidates}) == 1:
                    context.extend(candidates)
            if not context:
                continue
            # The narrowed descriptor is only a reader query. The runtime case
            # retains the original complete condition and all required roles.
            query = condition.model_copy(update={"roles": [role]})
            for material in case.evidence:
                selectors = [
                    o for o in context if o.evidence_id != material.evidence_id
                ]
                if not selectors:
                    continue
                source = reader.source(material)
                views = [
                    (view, None)
                    for view in source_views(
                        case,
                        material,
                        reader.encoder.processor.tokenizer,
                        config["reader"]["text_chunk_tokens"],
                    )
                ]
                views.extend(
                    image_views(
                        material,
                        source,
                        reader.output_root / "page_views",
                        pdf_dpi=config["reader"].get("pdf_dpi", 120),
                    )
                )
                selected = []
                for view, image in views:
                    try:
                        result = reader.read_many_roles(
                            case,
                            material,
                            [query],
                            view,
                            image=image,
                            query_context=selectors,
                        )
                        checked, failures = check_reads(
                            case, [o.model_dump() for o in result.operands]
                        )
                        selected.extend(checked)
                        issues.extend(failures)
                        issues.extend(
                            f"selection:{condition.condition_id}:{s}"
                            for s in result.issues
                        )
                    except (ValueError, RuntimeError) as error:
                        issues.append(
                            f"selection_failed:{condition.condition_id}:{material.evidence_id}:{error}"
                        )
                usable = [o for o in selected if o.status == "obtained"]
                if len({value_key(o) for o in usable}) == 1:
                    final = [
                        o
                        for o in final
                        if not (
                            o.condition_id == condition.condition_id
                            and o.role == role
                            and o.evidence_id == material.evidence_id
                        )
                    ]
                    final.extend(usable)
    raw = case.model_dump()
    raw["operands"] = list(
        {o.model_dump_json(): o.model_dump() for o in final}.values()
    )
    return AuditCase.model_validate(raw), issues
