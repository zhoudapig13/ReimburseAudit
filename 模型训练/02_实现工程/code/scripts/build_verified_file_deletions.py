"""Create genuine training-input file deletions with source-bound supervision."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from audit_engine.contracts import AuditCase, ConditionAnnotation
from audit_engine.inference.runner import scalar_policies
from audit_engine.training.patent import read_jsonl, validate_annotations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant-prefix", default="patent_variants")
    args = p.parse_args()
    if Path(args.variant_prefix).name != args.variant_prefix or args.variant_prefix in {"", ".", ".."}:
        p.error("variant prefix must be a single directory name")
    args.output.mkdir(parents=True, exist_ok=False)
    original = [AuditCase.model_validate(row) for row in read_jsonl(args.input_run / "reader_cases.jsonl")]
    annotations = [ConditionAnnotation.model_validate(row) for row in read_jsonl(args.input_run / "annotations.jsonl")]
    partitions = {row["workflow_id"]: row for row in read_jsonl(Path("data/三模块交付_20261027/dataset/partitions.jsonl"))}
    source_root = Path("data_raw/ReimburseBench").resolve()
    variants, labels, records = [], [], []
    for parent in original:
        if partitions[parent.workflow_id]["split"] != "train":
            continue
        before = scalar_policies(parent)
        all_active = torch.ones(len(parent.evidence), dtype=torch.bool)
        receipt_ids = {identifier for owner in parent.application.receipt_owners for identifier in owner.get("evidence_ids", [])}
        policy_ids = {ref.evidence_id for condition in parent.conditions for ref in condition.policy_refs}
        selected = None
        for material in parent.evidence:
            if material.evidence_id not in receipt_ids or material.evidence_id in policy_ids:
                continue
            active = all_active.clone()
            active[[e.evidence_id for e in parent.evidence].index(material.evidence_id)] = False
            affected = []
            for condition, policy in zip(parent.conditions, before, strict=True):
                old = policy.rebuild(all_active, 8)
                new = policy.rebuild(active, 8)
                was_complete = bool(old[3].any())
                missing_role = bool((new[1] & ~new[0].any(-1)).any())
                if was_complete and missing_role:
                    affected.append(condition.condition_id)
            if affected:
                selected = material, affected
                break
        if selected is None:
            continue
        removed, affected = selected
        raw = parent.model_dump()
        identifier = parent.workflow_id + "_drop_" + removed.evidence_id
        folder = source_root / args.variant_prefix / identifier
        folder.mkdir(parents=True, exist_ok=False)
        raw.update(workflow_id=identifier)
        raw["evidence"] = [row for row in raw["evidence"] if row["evidence_id"] != removed.evidence_id]
        for material in raw["evidence"]:
            source = source_root / material["raw_ref"]
            target = folder / material["evidence_id"] / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if hashlib.sha256(target.read_bytes()).hexdigest() != material["source_sha256"]:
                raise ValueError("variant copy changed a retained original material")
            material["raw_ref"] = str(target.relative_to(source_root))
        raw["application"]["attachment_ids"].remove(removed.evidence_id)
        for owner in raw["application"]["receipt_owners"]:
            owner["evidence_ids"] = [key for key in owner.get("evidence_ids", []) if key != removed.evidence_id]
        raw["application"]["receipt_owners"] = [owner for owner in raw["application"]["receipt_owners"] if owner["evidence_ids"]]
        surviving = []
        for observation in raw["operands"]:
            if observation["evidence_id"] == removed.evidence_id:
                continue
            groups = observation["dependency_groups"] or [[key] for key in observation["depends_on"]]
            groups = [[key for key in group if key != removed.evidence_id] for group in groups]
            if any(not group for group in groups):
                continue
            observation["depends_on"] = sorted({key for group in groups for key in group})
            observation["dependency_groups"] = groups
            surviving.append(observation)
        raw["operands"] = surviving
        source_record = {"parent": parent.workflow_id, "removed_evidence_id": removed.evidence_id,
                         "removed_source_sha256": removed.source_sha256, "affected_conditions": affected,
                         "retained_originals_copied_and_hash_verified": True,
                         "deleted_file_present_in_variant": False,
                         "unaffected_reads_reused_only_for_identical_source_hashes": True}
        version = hashlib.sha256(json.dumps(source_record, sort_keys=True).encode()).hexdigest()
        raw["input_version"] = "patent_reader_v1:verified_source_deletion:" + version
        variant = AuditCase.model_validate(raw)
        rebuilt = scalar_policies(variant)
        active = torch.ones(len(variant.evidence), dtype=torch.bool)
        for condition, policy in zip(variant.conditions, rebuilt, strict=True):
            if condition.condition_id in affected:
                role_map, required, _, _ = policy.rebuild(active, 8)
                for role_index in (required & ~role_map.any(-1)).nonzero().flatten().tolist():
                    raw["unresolved"].append(f"confirmed_missing:{condition.condition_id}:{condition.roles[role_index]}")
        variant = AuditCase.model_validate(raw)
        known = {material.evidence_id for material in variant.evidence}
        for annotation in annotations:
            if annotation.workflow_id != parent.workflow_id:
                continue
            label = copy.deepcopy(annotation.model_dump())
            label.update(workflow_id=identifier, augmentation_of=parent.workflow_id,
                         case_sha256=variant.digest(), method="verified_original_file_deletion:" + version,
                         reviewer="codex_source_inspection",
                         review_reason="Actual input copy excludes the removed file. Identical remaining source hashes retain source reads. A formerly executable condition now lacks a necessary source role; affected whole-condition relations become unknown, with no positive polarity citation.")
            label["relations"] = [row for row in label["relations"] if row["evidence_id"] in known]
            label["references"] = [row for row in label["references"] if row["evidence_id"] in known]
            if annotation.condition_id in affected:
                label["state"] = "insufficient"
                for row in label["relations"]:
                    row["relation"] = "unknown"
                for row in label["references"]:
                    row["label"] = 0
            labels.append(ConditionAnnotation.model_validate(label))
        partitions[identifier] = {**partitions[parent.workflow_id], "workflow_id": identifier}
        source_record.update(workflow_id=identifier, case_sha256=variant.digest(), source_family_id=partitions[identifier]["source_family_id"])
        (folder / "deletion_manifest.json").write_text(json.dumps(source_record, ensure_ascii=False, indent=2))
        variants.append(variant)
        records.append(source_record)
    cases = original + variants
    combined = annotations + labels
    validate_annotations(cases, combined, partitions)
    for name, rows in [("reader_cases.jsonl", [case.model_dump() for case in cases]),
                       ("annotations.jsonl", [row.model_dump() for row in combined]),
                       ("partitions.jsonl", list(partitions.values()))]:
        (args.output / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    report = {"variants": records, "variant_count": len(variants), "original_workflows": len(original),
              "only_training_split_augmented": True, "reference_answers_read": False,
              "factual_change": "actual_input_material_file_deletion"}
    (args.output / "augmentation_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "variants"}), flush=True)


if __name__ == "__main__":
    main()
