"""Local, resumable source reading and training-only label proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import yaml

from audit_engine.contracts import AuditCase
from audit_engine.readers.absence import bind_explicit_date_absence
from audit_engine.readers.material_views import image_views
from audit_engine.readers.patent_reader import PatentReader, source_views
from audit_engine.readers.qwen_encoder import FrozenQwenEncoder
from audit_engine.readers.selection import check_reads, resolve_selection_reads
from audit_engine.readers.table_sets import bind_identifier_roles
from audit_engine.training.patent import read_jsonl


def select_pilot(cases, partitions, split, count):
    by_family = defaultdict(list)
    for case in cases:
        partition = partitions[case.workflow_id]
        if partition["split"] == split:
            by_family[partition["source_family_id"]].append(case)
    for group in by_family.values():
        group.sort(
            key=lambda c: (sum(len(e.text or "") for e in c.evidence), c.workflow_id)
        )
    groups = sorted(
        by_family.values(),
        key=lambda g: (sum(len(e.text or "") for e in g[0].evidence), g[0].workflow_id),
    )
    ordered = [
        group[i]
        for i in range(max(map(len, groups), default=0))
        for group in groups
        if i < len(group)
    ]
    return ordered[:count] if count else ordered


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def process_case(reader, case, mode, config, partition, reference=None):
    candidate = case.model_dump()
    issues = []
    source_column_reads = []
    source_absence_reads = []
    if mode == "propose":
        # Candidate rules have application scope until source review allocates
        # them to disjoint receipts. Retaining item-specific amount programs
        # here creates dangling bindings to the replaced condition set.
        candidate["amount_units"] = [{"object_id": "claim", "scope": "application",
            "claimed_amount": case.application.claimed_amount,
            "currency": case.application.currency, "limit_status": "unknown"}]
        candidate["amount_programs"] = []
        candidate["policy_numbers"] = []
        candidate["policy_coverage"] = "pending"
        conditions = {}
        for material in case.evidence:
            query_material = material
            filename = Path(material.raw_ref).name
            if re.search(r"exchange[_ ]rate[_ ]reference", filename, re.IGNORECASE):
                issues.append(f"policy_source_index:{material.evidence_id}:currency_rate_reference_retained_for_amount_reading")
                continue
            if material.type == "xlsx" and not re.search(r"standard|policy|approval|contract", filename, re.IGNORECASE):
                issues.append(f"policy_source_index:{material.evidence_id}:factual_record_retained_as_evidence")
                continue
            if material.type == "xlsx" and re.search(r"reimbursement[_ ]standard", filename, re.IGNORECASE):
                application = case.application.application.casefold()
                category = "meal" if re.search(r"meal|refreshment|food", application) else "transport" if re.search(r"transport|travel|train|commut|bus|taxi|flight", application) else None
                if category:
                    sheets = {}
                    for segment in material.segments:
                        match = re.match(r"Sheet ([^:]+):", segment.text)
                        if match:
                            sheets.setdefault(match.group(1), []).append(segment)
                    pattern = r"meal|food" if category == "meal" else r"transport"
                    selected = [segment for sheet, segments in sheets.items()
                                if re.search(pattern, sheet, re.IGNORECASE) for segment in segments]
                    if selected:
                        query_material = material.model_copy(update={"segments": selected})
                        issues.append(f"policy_source_index:{material.evidence_id}:{category}:selected_{len(selected)}_of_{len(material.segments)}_rows;all_original_rows_retained_in_evidence")
            for view in source_views(
                case,
                query_material,
                reader.encoder.processor.tokenizer,
                config["reader"]["text_chunk_tokens"],
            ):
                try:
                    result = reader.propose(case, material, view)
                    issues.extend(result.issues)
                    for condition in result.conditions:
                        raw = condition.model_dump()
                        raw.pop("condition_id")
                        key = hashlib.sha256(
                            json.dumps(raw, sort_keys=True).encode()
                        ).hexdigest()[:16]
                        raw["condition_id"] = "c_" + key
                        conditions[key] = raw
                except (ValueError, RuntimeError) as error:
                    issues.append(f"policy_read:{material.evidence_id}:{error}")
        candidate["conditions"] = list(conditions.values())
        candidate["operands"] = []
        candidate["unresolved"] = sorted(
            set([*case.unresolved, "policy_candidates_require_review"])
        )
        case = AuditCase.model_validate(candidate)
    elif mode == "read":
        operands, completed = [], set()
        for material in case.evidence:
            source = reader.source(material)
            views = [
                (v, None)
                for v in source_views(
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
            success = True
            verified = [c for c in case.conditions if c.compile_status == "verified"]
            issues.extend(
                f"policy_review_required:{c.condition_id}"
                for c in case.conditions
                if c.compile_status != "verified"
            )
            batch_size = config["reader"].get("condition_batch", 3)
            if batch_size < 1:
                raise ValueError("condition_batch must be positive")
            for start in range(0, len(verified), batch_size):
                batch = verified[start : start + batch_size]
                for view, image in views:
                    try:
                        result = reader.read_many_roles(
                            case, material, batch, view, image=image
                        )
                        operands.extend(o.model_dump() for o in result.operands)
                        issues.extend(
                            f"read:{material.evidence_id}:{s}" for s in result.issues
                        )
                    except (ValueError, RuntimeError) as error:
                        try:
                            result = reader.read_many_roles(case, material, batch, view, image=image, reread_reason=str(error))
                            operands.extend(o.model_dump() for o in result.operands)
                            issues.extend(f"read:{material.evidence_id}:{s}" for s in result.issues)
                            issues.append(f"source_reread_recovered:{material.evidence_id}:{','.join(c.condition_id for c in batch)}")
                        except (ValueError, RuntimeError) as repeated:
                            success = False
                            issues.append(f"read_failed:{material.evidence_id}:{','.join(c.condition_id for c in batch)}:{repeated}")
            if verified and views and success:
                completed.add(material.evidence_id)
            for evidence in candidate["evidence"]:
                if evidence["evidence_id"] == material.evidence_id:
                    evidence["read_status"] = (
                        "obtained" if material.evidence_id in completed else "partial"
                    )
        # Identical repeated observations are one read; contradictory reads remain
        # explicit and make the runtime operand resolution ambiguous.
        candidate["operands"] = list(
            {json.dumps(o, sort_keys=True): o for o in operands}.values()
        )
        structural = case.model_copy(
            update={
                "evidence": [
                    type(e).model_validate(raw)
                    for e, raw in zip(case.evidence, candidate["evidence"], strict=True)
                ]
            }
        )
        checked, failures = check_reads(structural, candidate["operands"])
        issues.extend(failures)
        candidate["operands"] = [o.model_dump() for o in checked]
        working = AuditCase.model_validate(candidate)
        working, selection_issues = resolve_selection_reads(reader, working, config)
        issues.extend(selection_issues)
        working, source_column_reads = bind_identifier_roles(working, reader.source_root)
        working, source_absence_reads = bind_explicit_date_absence(working, reader.source_root)
        candidate["operands"] = [o.model_dump() for o in working.operands]
        call_hash = hashlib.sha256(
            json.dumps({"calls": sorted(e["record"] for e in reader.events),
                        "original_columns": source_column_reads,
                        "source_absence": source_absence_reads}, sort_keys=True).encode()
        ).hexdigest()
        candidate["input_version"] = "patent_reader_v1:" + call_hash
        candidate["unresolved"] = [
            s
            for s in working.unresolved
            if not any(s == "material_reading_required:" + key for key in completed)
        ]
        if any(s.startswith("read_failed:") for s in issues):
            candidate["unresolved"].append("material_reader_failure")
        for issue in issues:
            match = re.search(r"confirmed_absent_field:(e\d+):([^:]+):([^:]+)$", issue)
            if not match:
                continue
            evidence_id, condition_id, role = match.groups()
            observations = [o for o in working.operands if o.condition_id == condition_id and o.role == role]
            if not any(o.status == "obtained" for o in observations):
                candidate["unresolved"].append(f"confirmed_missing:{condition_id}:{role}")
        candidate["unresolved"] = sorted(set(candidate["unresolved"]))
        case = AuditCase.model_validate(candidate)
    else:
        if not case.input_version.startswith("patent_reader_v1:"):
            raise ValueError("teacher candidates require actual source-reader inputs")
        annotations = []
        for condition in case.conditions:
            try:
                result = reader.teacher(case, condition, partition, reference)
                annotations.extend(a.model_dump() for a in result.conditions)
                issues.extend(result.annotation_issues)
            except (ValueError, RuntimeError) as error:
                issues.append(f"teacher:{condition.condition_id}:{error}")
        return {
            "case_sha256": case.digest(),
            "workflow_id": case.workflow_id,
            "annotations": annotations,
            "annotation_issues": issues,
            "status": "pending_review",
            "reader_events": list(reader.events),
        }
    return {
        "case": case.model_dump(),
        "case_sha256": case.digest(),
        "workflow_id": case.workflow_id,
        "issues": issues,
        "status": "pending_review",
        "reader_events": list(reader.events),
        "source_column_reads": source_column_reads,
        "source_absence_reads": source_absence_reads,
        "proposal_amount_scope": "aggregate_placeholder_pending_source_allocation" if mode == "propose" else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["propose", "read", "teacher"], required=True)
    p.add_argument("--cases", type=Path, default=Path("data/三模块交付_20261027/dataset/reader_cases.jsonl"))
    p.add_argument(
        "--partitions", type=Path, default=Path("data/三模块交付_20261027/dataset/partitions.jsonl")
    )
    p.add_argument("--split", choices=["train", "validation", "test"], default="train")
    p.add_argument("--pilot-count", type=int, default=24)
    p.add_argument("--workflow-list", type=Path)
    p.add_argument("--source-root", type=Path, default=Path("data_raw/ReimburseBench"))
    p.add_argument("--config", type=Path, default=Path("configs/patent_delivery.yaml"))
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--references",
        type=Path,
        default=Path("data/labels_private/train_reference_targets.jsonl"),
    )
    args = p.parse_args()
    contract_files = [Path(__file__), Path("audit_engine/readers/patent_reader.py"),
                      Path("audit_engine/readers/value_formats.py"),
                      Path("audit_engine/readers/literal_bindings.py"),
                      Path("audit_engine/readers/selection.py"),
                      Path("audit_engine/readers/table_sets.py"),
                      Path("audit_engine/readers/absence.py"),
                      Path("audit_engine/readers/material_views.py"),
                      Path("audit_engine/readers/qwen_encoder.py"),
                      Path("audit_engine/contracts.py")]
    reader_contract = hashlib.sha256(b"".join(path.read_bytes() for path in contract_files)).hexdigest()
    if args.mode == "teacher" and args.split != "train":
        p.error("reference-assisted teacher candidates are training-only")
    index, count = map(int, args.shard.split("/"))
    if not 0 <= index < count or args.pilot_count < 0:
        p.error("invalid shard or pilot count")
    config = yaml.safe_load(args.config.read_text())
    partitions = {v["workflow_id"]: v for v in read_jsonl(args.partitions)}
    cases = [AuditCase.model_validate(v) for v in read_jsonl(args.cases)]
    selected = select_pilot(cases, partitions, args.split, args.pilot_count)
    if args.workflow_list:
        ids = json.loads(args.workflow_list.read_text())["workflow_ids"]
        by_id = {c.workflow_id: c for c in cases}
        if len(ids) != len(set(ids)) or any(
            key not in by_id or partitions[key]["split"] != args.split for key in ids
        ):
            raise ValueError("explicit workflow selection differs from the fixed split")
        selected = [by_id[key] for key in ids]
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / f"shard_{index:04d}"
    output.mkdir(exist_ok=True)
    save(
        output / "selection.json",
        {
            "workflows": [c.workflow_id for c in selected],
            "shard": args.shard,
            "source_families": sorted(
                {partitions[c.workflow_id]["source_family_id"] for c in selected}
            ),
            "mode": args.mode,
            "split": args.split,
        },
    )
    references = (
        {v["workflow_id"]: v["target"] for v in read_jsonl(args.references)}
        if args.mode == "teacher"
        else {}
    )
    torch.set_num_threads(4)
    encoder = FrozenQwenEncoder(config, args.model_path, args.device)
    reader = PatentReader(encoder, args.source_root, output / "reader_calls")
    for case in selected[index::count]:
        record = output / (
            hashlib.sha256(case.workflow_id.encode()).hexdigest() + ".json"
        )
        if record.is_file():
            saved = json.loads(record.read_text())
            if (
                saved.get("source_case_sha256") == case.digest()
                and saved.get("model_signature") == encoder.signature
                and saved.get("reader_contract_sha256") == reader_contract
            ):
                print(
                    json.dumps({"workflow_id": case.workflow_id, "status": "resumed"}),
                    flush=True,
                )
                continue
            raise ValueError(
                "completed reader record has a different source or model version"
            )
        reader.events = []
        try:
            result = process_case(
                reader,
                case,
                args.mode,
                config,
                partitions[case.workflow_id],
                references.get(case.workflow_id),
            )
        except (ValueError, RuntimeError) as error:
            result = {
                "workflow_id": case.workflow_id,
                "status": "failed",
                "error": str(error),
            }
        result.update(
            source_case_sha256=case.digest(), model_signature=encoder.signature,
            reader_contract_sha256=reader_contract,
        )
        save(record, result)
        print(
            json.dumps(
                {
                    "workflow_id": case.workflow_id,
                    "status": result["status"],
                    "conditions": len(result.get("case", {}).get("conditions", [])),
                    "issues": len(
                        result.get("issues", result.get("annotation_issues", []))
                    ),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
