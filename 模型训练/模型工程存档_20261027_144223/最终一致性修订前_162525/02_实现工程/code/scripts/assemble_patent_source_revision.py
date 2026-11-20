"""Version actual source rereads and rebind encodings only if encoder inputs match."""

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from audit_engine.contracts import AuditCase
from audit_engine.readers.absence import bind_explicit_date_absence
from audit_engine.training.patent import feature_path, load_features, read_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--reads", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bind-explicit-absence", action="store_true")
    p.add_argument("--source-root", type=Path, default=Path("data_raw/ReimburseBench"))
    args = p.parse_args()
    source, output = args.source_run, args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / "dataset").mkdir()
    (output / "features").mkdir()
    records = {}
    for path in args.reads.glob("shard_*/*.json"):
        if path.name == "selection.json":
            continue
        row = json.loads(path.read_text())
        if row.get("status") != "complete" or row.get("reference_answers_read") is not False:
            raise ValueError("incomplete or reference-assisted source record")
        if row["workflow_id"] in records:
            raise ValueError("duplicate source reread")
        records[row["workflow_id"]] = row
    cases, changes, absence_proofs = {}, [], []
    for row in read_jsonl(source / "dataset/reader_cases.jsonl"):
        old = AuditCase.model_validate(row)
        if old.workflow_id in records:
            record = records[old.workflow_id]
            if record["input_sha256"] != old.digest():
                raise ValueError("source reread refers to a different input version")
            updated = AuditCase.model_validate(record["case"])
            replaced = record["replaced_roles"]
        elif "_drop_" in old.workflow_id and old.workflow_id.split("_drop_", 1)[0] in records:
            record = records[old.workflow_id.split("_drop_", 1)[0]]
            parent = AuditCase.model_validate(record["case"])
            keys = {tuple(key) for key in record["replaced_roles"]}
            existing = {e.evidence_id for e in old.evidence}
            conditions = {c.condition_id for c in old.conditions}
            reads = [o for o in parent.operands
                     if (o.condition_id, o.role, o.evidence_id) in keys
                     and set(o.depends_on).issubset(existing) and o.condition_id in conditions]
            inherited = {(o.condition_id, o.role, o.evidence_id) for o in reads}
            raw = old.model_dump()
            raw["operands"] = [o for o in raw["operands"]
                               if (o["condition_id"], o["role"], o["evidence_id"]) not in inherited]
            raw["operands"].extend(o.model_dump() for o in reads)
            if inherited:
                raw["input_version"] = "patent_reader_v1:inherited_source_reread:" + hashlib.sha256(
                    json.dumps(raw["operands"], sort_keys=True).encode()).hexdigest()
            updated, replaced = AuditCase.model_validate(raw), sorted(inherited)
        else:
            updated, replaced = old, []
        if args.bind_explicit_absence:
            updated, proofs = bind_explicit_date_absence(updated, args.source_root)
            absence_proofs.extend(dict(proof, workflow_id=updated.workflow_id) for proof in proofs)
            replaced = list(replaced) + [(proof["condition_id"], proof["role"], proof["evidence_id"]) for proof in proofs]
        if (old.application, old.evidence, old.conditions) != (
            updated.application, updated.evidence, updated.conditions
        ):
            raise ValueError("source reread changed encoder inputs; fresh encoding is required")
        feature = load_features(feature_path(source / "features", old.workflow_id))
        feature.validate(old, feature.application.numel())
        feature.case_sha256 = updated.digest()
        feature.validate(updated, feature.application.numel())
        torch.save(asdict(feature), feature_path(output / "features", old.workflow_id))
        cases[updated.workflow_id] = updated
        if replaced:
            changes.append({"workflow_id": updated.workflow_id, "replaced_roles": replaced,
                            "old_case_sha256": old.digest(), "new_case_sha256": updated.digest()})
    (output / "dataset/reader_cases.jsonl").write_text("".join(c.model_dump_json() + "\n" for c in cases.values()))
    labels = list(read_jsonl(source / "dataset/annotations.jsonl"))
    for label in labels:
        label["case_sha256"] = cases[label["workflow_id"]].digest()
    (output / "dataset/annotations.jsonl").write_text("".join(json.dumps(label, ensure_ascii=False) + "\n" for label in labels))
    shutil.copyfile(source / "dataset/partitions.jsonl", output / "dataset/partitions.jsonl")
    identifiers = [r["workflow_id"] for r in read_jsonl(source / "validation_cases.jsonl")]
    (output / "validation_cases.jsonl").write_text("".join(cases[key].model_dump_json() + "\n" for key in identifiers))
    report = {"source_run": str(source), "reads": str(args.reads), "encoder_inputs_changed": False,
              "policy_coverage_changed": False, "reference_answers_read": False, "changes": changes}
    report["explicit_absence_bindings"] = absence_proofs
    (output / "source_revision.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"changed_workflows": len(changes), "changed_roles": sum(len(c["replaced_roles"]) for c in changes)}))


if __name__ == "__main__":
    main()
