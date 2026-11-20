"""Target missing source-bound roles; retain actual calls and uncertainty."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml

from audit_engine.contracts import AuditCase
from audit_engine.readers.material_views import image_views
from audit_engine.readers.patent_reader import (
    PatentReader,
    declared_categories,
    source_views,
)
from audit_engine.readers.qwen_encoder import FrozenQwenEncoder
from audit_engine.readers.selection import check_reads, value_key
from audit_engine.readers.value_formats import missing_required_field
from audit_engine.training.patent import read_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--partitions", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--source-root", type=Path, default=Path("data_raw/ReimburseBench"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--retry-incomplete", action="store_true")
    p.add_argument("--categories-only", action="store_true")
    p.add_argument("--inconsistencies-only", action="store_true")
    args = p.parse_args()
    parts = {r["workflow_id"]: r for r in read_jsonl(args.partitions)}
    shard, shards = map(int, args.shard.split("/"))
    if not 0 <= shard < shards:
        p.error("invalid shard")
    config = yaml.safe_load(args.config.read_text())
    encoder = FrozenQwenEncoder(config, args.model_path, args.device)
    reader = PatentReader(encoder, args.source_root, args.output / "calls")
    args.output.mkdir(parents=True, exist_ok=True)
    selected = []
    for row in read_jsonl(args.cases):
        case = AuditCase.model_validate(row)
        if parts[case.workflow_id]["split"] == "test" or "_drop_" in case.workflow_id:
            continue
        requests = {}
        for condition in case.conditions:
            for role in condition.roles:
                categories = declared_categories(condition.role_definitions.get(role, ""))
                if args.categories_only and not categories:
                    continue
                match = re.search(r"(?:^|_)(e\d+)(?:_|$)", role)
                if not match or role in condition.derived_roles:
                    continue
                observations = [o for o in case.operands if o.role == role
                                and o.condition_id == condition.condition_id and o.status == "obtained"]
                inconsistent = any(o.value == "complete" and missing_required_field(
                    " ".join(ref.note for ref in o.refs), condition.role_definitions.get(role, ""))
                    for o in observations)
                if args.inconsistencies_only and not inconsistent:
                    continue
                if (len({value_key(o) for o in observations}) == 1
                        and (not categories or all(o.value in categories for o in observations))
                        and not inconsistent):
                    continue
                requests.setdefault(match.group(1), {}).setdefault(condition.condition_id, []).append(role)
        if requests:
            selected.append((case, requests))
    expected = [case.workflow_id for case, _ in selected][shard::shards]
    (args.output / "selection.json").write_text(json.dumps({"workflow_ids": expected}, indent=2))
    for case, requests in selected[shard::shards]:
        target = args.output / (case.workflow_id + ".json")
        if target.exists():
            saved = json.loads(target.read_text())
            if (saved.get("input_sha256") == case.digest() and saved.get("status") == "complete"
                    and not (args.retry_incomplete and saved.get("issues"))):
                continue
        fresh, issues, replaced = [], [], []
        conditions = {c.condition_id: c for c in case.conditions}
        for evidence_id, groups in requests.items():
            material = next(e for e in case.evidence if e.evidence_id == evidence_id)
            queries = [conditions[key].model_copy(update={"roles": roles}) for key, roles in groups.items()]
            source = reader.source(material)
            views = [(view, None) for view in source_views(case, material, encoder.processor.tokenizer,
                                                          config["reader"]["text_chunk_tokens"])]
            views.extend(image_views(material, source, args.output / "pages",
                                     pdf_dpi=config["reader"].get("pdf_dpi", 120)))
            observations = []
            for view, image in views:
                reason = None
                for attempt in range(2):
                    try:
                        result = reader.read_many_roles(case, material, queries, view,
                                                        image=image, reread_reason=reason)
                        checked, errors = check_reads(case, [o.model_dump() for o in result.operands])
                        observations.extend(checked)
                        issues.extend(result.issues + errors)
                        break
                    except (ValueError, RuntimeError) as error:
                        reason = str(error)
                        if attempt:
                            issues.append(reason)
            for condition_id, roles in groups.items():
                for role in roles:
                    candidates = [o for o in observations if o.condition_id == condition_id
                                  and o.role == role and o.status == "obtained"]
                    if len({value_key(o) for o in candidates}) == 1:
                        replaced.append((condition_id, role, evidence_id))
                        fresh.extend(candidates)
        raw = case.model_dump()
        raw["operands"] = [o for o in raw["operands"]
                           if (o["condition_id"], o["role"], o["evidence_id"]) not in replaced]
        raw["operands"].extend(o.model_dump() for o in fresh)
        raw["input_version"] = "patent_reader_v1:targeted_source_reread:" + hashlib.sha256(
            json.dumps(raw["operands"], sort_keys=True).encode()).hexdigest()
        result = AuditCase.model_validate(raw)
        record = {"workflow_id": case.workflow_id, "status": "complete", "input_sha256": case.digest(),
                  "case": result.model_dump(), "replaced_roles": replaced, "issues": issues,
                  "reference_answers_read": False}
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(target)
        print(json.dumps({"workflow_id": case.workflow_id, "replaced_roles": len(replaced),
                          "issues": len(issues)}), flush=True)


if __name__ == "__main__":
    main()
