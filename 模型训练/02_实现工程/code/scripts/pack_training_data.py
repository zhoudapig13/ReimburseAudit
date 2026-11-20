"""Pack audited ReimburseBench workflows into separated input and target JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any

SPLIT_NAMES = ("train", "validation", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
KNOWN_REFERENCE_ISSUES = {
    "1.1.1": [
        "The guide flags a possible inconsistency between reference_max_legal=340 "
        "and the stated 220 USD/night for two nights; human review is required."
    ]
}
FORBIDDEN_INPUT_KEYS = {
    "difficulty",
    "expected_types",
    "reference_answer",
    "reference_outputs",
    "split",
    "target",
}


def json_text(value: Any, *, indent: int | None = None) -> str:
    """Serialize supported values without converting decimals through float."""
    return json.dumps(
        to_json_value(value),
        ensure_ascii=False,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    )


def to_json_value(value: Any) -> Any:
    """Convert Decimals recursively into exact decimal strings."""
    if isinstance(value, Decimal):
        return decimal_string(value)
    if isinstance(value, dict):
        return {str(key): to_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_json_value(item) for item in value]
    return value


def decimal_string(value: Any) -> str:
    """Return a finite, non-scientific decimal representation."""
    try:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"invalid decimal value: {value!r}") from error
    if not decimal_value.is_finite():
        raise ValueError(f"decimal value must be finite: {value!r}")
    return format(decimal_value, "f")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, parse_float=Decimal, parse_int=Decimal)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line, parse_float=Decimal, parse_int=Decimal)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            records.append(value)
    return records


def normalize_relative_path(value: Any) -> str:
    """Validate a workflow-local path and normalize separators to POSIX."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("attachment path must be a non-empty string")
    candidate = PurePosixPath(value.replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe attachment path: {value!r}")
    return candidate.as_posix()


def normalize_receipt_files(value: Any) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        raise TypeError("receipt mapping value must be a string or list")
    if not items:
        raise ValueError("receipt mapping list cannot be empty")
    return [normalize_relative_path(item) for item in items]


def evidence_type(extension: str) -> str:
    normalized = extension.lower()
    if normalized in IMAGE_EXTENSIONS:
        return "image"
    if normalized in {".pdf", ".docx", ".xlsx", ".json"}:
        return normalized.removeprefix(".")
    raise ValueError(f"unsupported evidence extension: {extension!r}")


def build_input_bundle(
    task: dict[str, Any],
    manifest_row: dict[str, Any],
    assignment: dict[str, Any],
) -> dict[str, Any]:
    """Map one audited workflow to a label-free canonical input bundle."""
    workflow_id = str(task.get("id", ""))
    if not workflow_id or workflow_id != str(manifest_row.get("workflow_id", "")):
        raise ValueError("task and manifest workflow IDs do not match")
    if workflow_id != str(assignment.get("workflow_id", "")):
        raise ValueError("task and split-assignment workflow IDs do not match")

    source_files = task.get("source_files")
    if not isinstance(source_files, dict):
        raise ValueError("task source_files must be an object")
    backgrounds = source_files.get("Background_information", [])
    receipts = source_files.get("Receipts", {})
    if not isinstance(backgrounds, list) or not isinstance(receipts, dict):
        raise ValueError("invalid Background_information or Receipts structure")

    manifest_files = {}
    for file_record in manifest_row.get("files", []):
        relative_path = normalize_relative_path(file_record.get("relative_path"))
        if relative_path in manifest_files:
            raise ValueError(
                f"duplicate manifest path in {workflow_id}: {relative_path}"
            )
        manifest_files[relative_path] = file_record

    ordered_paths: list[str] = []
    path_roles: dict[str, str] = {}
    owners_by_path: dict[str, list[str]] = {}
    for raw_path in backgrounds:
        relative_path = normalize_relative_path(raw_path)
        ordered_paths.append(relative_path)
        path_roles[relative_path] = "background"
        owners_by_path.setdefault(relative_path, [])

    owner_paths: list[tuple[str, list[str]]] = []
    for owner, raw_value in receipts.items():
        if owner == "standard":
            continue
        if not isinstance(owner, str) or not owner.strip():
            raise ValueError("receipt owner must be a non-empty string")
        receipt_paths = normalize_receipt_files(raw_value)
        owner_paths.append((owner, receipt_paths))
        for relative_path in receipt_paths:
            if relative_path not in path_roles:
                ordered_paths.append(relative_path)
            path_roles[relative_path] = "receipt"
            owners_by_path.setdefault(relative_path, []).append(owner)

    if len(ordered_paths) != len(set(ordered_paths)):
        raise ValueError(f"duplicate attachment reference in {workflow_id}")
    if set(ordered_paths) != set(manifest_files):
        missing = sorted(set(ordered_paths) - set(manifest_files))
        extra = sorted(set(manifest_files) - set(ordered_paths))
        raise ValueError(
            f"task/manifest attachment mismatch for {workflow_id}; "
            f"missing={missing}, extra={extra}"
        )

    evidence: list[dict[str, Any]] = []
    path_to_evidence_id: dict[str, str] = {}
    for index, relative_path in enumerate(ordered_paths, start=1):
        evidence_id = f"e{index:03d}"
        path_to_evidence_id[relative_path] = evidence_id
        file_record = manifest_files[relative_path]
        if not file_record.get("exists", False):
            raise ValueError(
                f"missing audited attachment: {workflow_id}/{relative_path}"
            )
        extension = str(file_record.get("extension", Path(relative_path).suffix))
        item_type = evidence_type(extension)
        views = (
            [
                {
                    "view_id": f"{evidence_id}:v001",
                    "source_id": evidence_id,
                    "kind": "original_image",
                }
            ]
            if item_type == "image"
            else []
        )
        evidence.append(
            {
                "evidence_id": evidence_id,
                "raw_ref": (
                    PurePosixPath("workflow") / workflow_id / relative_path
                ).as_posix(),
                "type": item_type,
                "role_hint": path_roles[relative_path],
                "owners": owners_by_path[relative_path],
                "text": None,
                "segments": [],
                "views": views,
                "parse_status": (
                    "image_available" if item_type == "image" else "source_available"
                ),
            }
        )

    receipt_owners = [
        {
            "owner": owner,
            "evidence_ids": [path_to_evidence_id[path] for path in paths],
        }
        for owner, paths in owner_paths
    ]
    claimed_amount = decimal_string(task.get("required_amount"))
    if Decimal(claimed_amount) < 0:
        raise ValueError(f"claimed amount must be non-negative: {workflow_id}")
    currency = str(task.get("currency", "")).strip()
    if not currency:
        raise ValueError(f"currency is missing: {workflow_id}")

    bundle = {
        "schema_version": "1.0",
        "input_version": "bundle_v1",
        "workflow_id": workflow_id,
        "source_family_id": str(assignment["source_family_id"]),
        "input": {
            "application": str(task.get("context", "")),
            "claimed_amount": claimed_amount,
            "currency": currency,
            "receipt_owners": receipt_owners,
            "attachment_ids": [item["evidence_id"] for item in evidence],
            "objects": [
                {
                    "object_id": "claim",
                    "scope": "application",
                    "claimed_amount": claimed_amount,
                }
            ],
        },
        "evidence": evidence,
    }
    issues = validate_input_bundle(bundle)
    if issues:
        raise ValueError(f"invalid input bundle for {workflow_id}: {issues}")
    return bundle


def nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(nested_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(nested_keys(item))
    return keys


def validate_input_bundle(bundle: dict[str, Any]) -> list[str]:
    """Validate references and reject label-bearing fields in model inputs."""
    issues: list[str] = []
    forbidden = sorted(nested_keys(bundle) & FORBIDDEN_INPUT_KEYS)
    if forbidden:
        issues.append(f"forbidden input keys: {forbidden}")

    evidence = bundle.get("evidence", [])
    evidence_ids = [item.get("evidence_id") for item in evidence]
    if len(evidence_ids) != len(set(evidence_ids)):
        issues.append("evidence IDs must be unique")
    attachment_ids = bundle.get("input", {}).get("attachment_ids", [])
    if attachment_ids != evidence_ids:
        issues.append("attachment IDs must match evidence order exactly")
    evidence_id_set = set(evidence_ids)
    for owner in bundle.get("input", {}).get("receipt_owners", []):
        dangling = set(owner.get("evidence_ids", [])) - evidence_id_set
        if dangling:
            issues.append(
                f"receipt owner has dangling evidence IDs: {sorted(dangling)}"
            )
    for item in evidence:
        raw_ref = str(item.get("raw_ref", ""))
        lowered = raw_ref.lower()
        if "reference_answer" in lowered or "/../" in f"/{raw_ref}/":
            issues.append(f"unsafe or answer-bearing raw_ref: {raw_ref}")
    try:
        claimed_amount = Decimal(str(bundle.get("input", {}).get("claimed_amount")))
        if not claimed_amount.is_finite() or claimed_amount < 0:
            issues.append("claimed amount must be finite and non-negative")
    except InvalidOperation:
        issues.append("claimed amount is not a decimal string")
    if not str(bundle.get("input", {}).get("currency", "")).strip():
        issues.append("currency must be non-empty")
    return issues


def normalize_flag(value: Any) -> int:
    if value in (Decimal(0), 0, False):
        return 0
    if value in (Decimal(1), 1, True):
        return 1
    raise ValueError(f"unsupported reference flag: {value!r}")


def build_reference_target(
    task: dict[str, Any],
    answer: dict[str, Any],
    manifest_row: dict[str, Any],
    assignment: dict[str, Any],
) -> dict[str, Any]:
    """Convert benchmark answers without claiming condition-level approval."""
    workflow_id = str(task.get("id", ""))
    if workflow_id != str(manifest_row.get("workflow_id", "")):
        raise ValueError("task and manifest workflow IDs do not match")
    if workflow_id != str(assignment.get("workflow_id", "")):
        raise ValueError("task and assignment workflow IDs do not match")

    flag_item = answer.get("Scoring_point_1_Flag")
    amount_item = answer.get("Scoring_point_2_Max_legal")
    if not isinstance(flag_item, dict) or not isinstance(amount_item, dict):
        raise ValueError(f"reference answer is missing flag or amount: {workflow_id}")

    scoring_points = []
    problem_reports = []
    score_sum = Decimal("0")
    for scoring_id, item in answer.items():
        if not isinstance(item, dict) or "answer" not in item or "score" not in item:
            raise ValueError(f"invalid scoring point {scoring_id!r}: {workflow_id}")
        score = Decimal(decimal_string(item["score"]))
        score_sum += score
        scoring_points.append(
            {
                "scoring_id": scoring_id,
                "answer": to_json_value(item["answer"]),
                "score": decimal_string(score),
            }
        )
        if scoring_id not in {"Scoring_point_1_Flag", "Scoring_point_2_Max_legal"}:
            problem_reports.append(str(item["answer"]))

    issues = list(KNOWN_REFERENCE_ISSUES.get(workflow_id, []))
    return {
        "schema_version": "1.0",
        "target_version": "benchmark_reference_v1",
        "input_version": "bundle_v1",
        "workflow_id": workflow_id,
        "source_family_id": str(assignment["source_family_id"]),
        "split": str(assignment["split"]),
        "target": {
            "reference_flag": normalize_flag(flag_item["answer"]),
            "reference_max_legal": decimal_string(amount_item["answer"]),
            "problem_reports": problem_reports,
            "logic_types": list(manifest_row.get("expected_types", [])),
            "scoring_points": scoring_points,
            "reference_score_sum": decimal_string(score_sum),
        },
        "annotation": {
            "status": "needs_review" if issues else "reference_only",
            "method": "benchmark_reference_conversion",
            "approved_for_component_training": False,
            "issues": issues,
        },
    }


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(content, encoding="utf-8", newline="\n")
    temporary_path.replace(path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    content = "".join(f"{json_text(record)}\n" for record in records)
    atomic_write_text(path, content)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# ReimburseBench 训练数据打包报告",
        "",
        "## 结论",
        "",
        f"- 已打包 {summary['workflow_count']} 条输入，输入与参考目标物理分离。",
        f"- 输入附件引用共 {summary['evidence_reference_count']} 条。",
        f"- 输入答案泄漏检查：{summary['input_leak_count']} 个问题。",
        "- 本包保存原始整单参考目标；条件级监督经原材料和政策审核后独立保存。",
        "",
        "## 分区",
        "",
        "| 分区 | 输入数 | 私有参考目标数 | 待复核目标数 |",
        "|---|---:|---:|---:|",
    ]
    for split_name in SPLIT_NAMES:
        item = summary["splits"][split_name]
        lines.append(
            f"| {split_name} | {item['input_count']} | "
            f"{item['target_count']} | {item['needs_review_count']} |"
        )
    lines.extend(
        [
            "",
            "## 数据用途",
            "",
            "- `data/packed/*_inputs.jsonl` 只包含申请、金额、币种、附件引用和中性证据编号。",
            "- `data/labels_private/*_reference_targets.jsonl` 保存原基准答案及逻辑类型。",
            "- `reference_only` 保存原始参考目标；条件关系与引用监督使用经核验的条件级标签及有效掩码。",
            "- 材料读取SFT问答由 `build_reader_qa.py` 基于解析材料和经核验的金额、日期、人员等事实构建。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--packed-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = datetime.now(timezone.utc)
    manifest_rows = {str(row["workflow_id"]): row for row in load_jsonl(args.manifest)}
    assignments = {str(row["workflow_id"]): row for row in load_jsonl(args.assignments)}
    if set(manifest_rows) != set(assignments):
        raise ValueError("manifest and split assignments cover different workflows")

    inputs_by_split = {split_name: [] for split_name in SPLIT_NAMES}
    targets_by_split = {split_name: [] for split_name in SPLIT_NAMES}
    input_issues: list[str] = []
    for workflow_id in sorted(
        manifest_rows, key=lambda value: tuple(map(int, value.split(".")))
    ):
        manifest_row = manifest_rows[workflow_id]
        assignment = assignments[workflow_id]
        split_name = str(assignment.get("split", ""))
        if split_name not in SPLIT_NAMES:
            raise ValueError(f"invalid split for {workflow_id}: {split_name!r}")
        workflow_dir = args.repo_root / "workflow" / workflow_id
        task = load_json(workflow_dir / "task_description.json")
        answer_names = task.get("reference_outputs", [])
        if answer_names != ["reference_answer.json"]:
            raise ValueError(
                f"unexpected reference_outputs for {workflow_id}: {answer_names!r}"
            )
        answer = load_json(workflow_dir / "reference_answer.json")
        input_bundle = build_input_bundle(task, manifest_row, assignment)
        input_issues.extend(
            f"{workflow_id}: {issue}" for issue in validate_input_bundle(input_bundle)
        )
        inputs_by_split[split_name].append(input_bundle)
        targets_by_split[split_name].append(
            build_reference_target(task, answer, manifest_row, assignment)
        )

    output_paths: list[Path] = []
    for split_name in SPLIT_NAMES:
        input_path = args.packed_dir / f"{split_name}_inputs.jsonl"
        target_path = args.labels_dir / f"{split_name}_reference_targets.jsonl"
        write_jsonl(input_path, inputs_by_split[split_name])
        write_jsonl(target_path, targets_by_split[split_name])
        output_paths.extend([input_path, target_path])

    summary = {
        "schema_version": "1.0",
        "status": "passed" if not input_issues else "failed",
        "workflow_count": sum(len(items) for items in inputs_by_split.values()),
        "evidence_reference_count": sum(
            len(bundle["evidence"])
            for records in inputs_by_split.values()
            for bundle in records
        ),
        "input_leak_count": len(input_issues),
        "input_issues": input_issues,
        "component_training_approved_count": 0,
        "splits": {
            split_name: {
                "input_count": len(inputs_by_split[split_name]),
                "target_count": len(targets_by_split[split_name]),
                "needs_review_count": sum(
                    target["annotation"]["status"] == "needs_review"
                    for target in targets_by_split[split_name]
                ),
                "reference_flag": dict(
                    sorted(
                        Counter(
                            str(target["target"]["reference_flag"])
                            for target in targets_by_split[split_name]
                        ).items()
                    )
                ),
            }
            for split_name in SPLIT_NAMES
        },
    }
    report_path = args.packed_dir / "pack_report.md"
    summary_path = args.packed_dir / "pack_summary.json"
    atomic_write_text(report_path, render_report(summary))
    atomic_write_text(summary_path, f"{json_text(summary, indent=2)}\n")
    output_paths.extend([report_path, summary_path])

    finished_at = datetime.now(timezone.utc)
    stage_manifest = {
        "schema_version": "1.0",
        "stage": "pack_training_data",
        "status": summary["status"],
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "inputs": {
            "manifest": str(args.manifest),
            "manifest_sha256": sha256_file(args.manifest),
            "assignments": str(args.assignments),
            "assignments_sha256": sha256_file(args.assignments),
            "repo_root": str(args.repo_root),
        },
        "configuration": {
            "input_version": "bundle_v1",
            "target_version": "benchmark_reference_v1",
            "model_version": None,
        },
        "counts": {
            "success": summary["workflow_count"],
            "failed": len(input_issues),
            "pending_review": sum(
                item["needs_review_count"] for item in summary["splits"].values()
            ),
        },
        "outputs": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in output_paths
        ],
    }
    atomic_write_text(
        args.packed_dir / "stage_manifest.json",
        f"{json_text(stage_manifest, indent=2)}\n",
    )
    print(json_text(summary, indent=2))
    return 0 if not input_issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
