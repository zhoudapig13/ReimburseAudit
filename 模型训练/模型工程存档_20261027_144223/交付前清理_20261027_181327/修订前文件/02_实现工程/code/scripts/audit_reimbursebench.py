"""Audit the raw ReimburseBench release without modifying source files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import zipfile
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

TASK_FILENAME = "task_description.json"
METADATA_FILENAME = "metadata.json"
SUPPORTED_INPUT_EXTENSIONS = {
    ".docx",
    ".jpeg",
    ".jpg",
    ".json",
    ".pdf",
    ".png",
    ".xlsx",
}
FLAG_PREFIX = "Scoring_point_1"
AMOUNT_PREFIX = "Scoring_point_2"


def resolve_reference(
    workflow_dir: Path, reference: str
) -> tuple[Path | None, str | None]:
    """Resolve a workflow-local reference and reject absolute or escaping paths."""
    if not isinstance(reference, str) or not reference.strip():
        return None, "reference must be a non-empty string"

    reference_path = Path(reference)
    if reference_path.is_absolute():
        return None, "absolute paths are not allowed"

    workflow_root = workflow_dir.resolve()
    resolved = (workflow_dir / reference_path).resolve()
    try:
        resolved.relative_to(workflow_root)
    except ValueError:
        return None, "reference escapes workflow directory"
    return resolved, None


def normalize_receipt_value(value: Any) -> list[str]:
    """Normalize one receipt mapping value to a validated filename list."""
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        raise TypeError(
            f"receipt value must be a string or list, got {type(value).__name__}"
        )

    if not values or any(
        not isinstance(item, str) or not item.strip() for item in values
    ):
        raise TypeError("receipt filenames must be non-empty strings")
    return values


def normalize_flag(value: Any) -> int | None:
    """Normalize common binary reference-answer encodings."""
    if value in (0, 1, False, True):
        return int(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "is"}:
        return 1
    if normalized in {"0", "false", "no", "n", "non"}:
        return 0
    return None


def decimal_value(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream, parse_float=Decimal)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_zip_document(path: Path, required_member: str) -> tuple[str, str | None]:
    try:
        with zipfile.ZipFile(path) as archive:
            if required_member not in archive.namelist():
                return "failed", f"missing archive member {required_member}"
            broken_member = archive.testzip()
            if broken_member:
                return "failed", f"corrupt archive member {broken_member}"
    except (OSError, zipfile.BadZipFile) as error:
        return "failed", str(error)
    return "passed", None


def validate_pdf(path: Path) -> tuple[str, str | None]:
    try:
        with path.open("rb") as stream:
            header = stream.read(8)
            if not header.startswith(b"%PDF-"):
                return "failed", "missing PDF signature"
            stream.seek(max(0, path.stat().st_size - 4096))
            trailer = stream.read()
            if b"%%EOF" not in trailer:
                return "failed", "missing PDF EOF marker"
    except OSError as error:
        return "failed", str(error)
    return "passed", None


def validate_image(path: Path) -> tuple[str, str | None]:
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError:
        return "not_checked", "Pillow is not installed"

    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as error:
        return "failed", str(error)
    return "passed", None


def validate_json_file(path: Path) -> tuple[str, str | None]:
    try:
        load_json(path)
    except (OSError, json.JSONDecodeError) as error:
        return "failed", str(error)
    return "passed", None


def validate_file(path: Path) -> tuple[str, str | None]:
    if path.stat().st_size == 0:
        return "failed", "file is empty"
    extension = path.suffix.lower()
    if extension == ".docx":
        return validate_zip_document(path, "word/document.xml")
    if extension == ".xlsx":
        return validate_zip_document(path, "xl/workbook.xml")
    if extension == ".pdf":
        return validate_pdf(path)
    if extension in {".jpg", ".jpeg", ".png"}:
        return validate_image(path)
    if extension == ".json":
        return validate_json_file(path)
    return "not_checked", f"no integrity checker for {extension or '[no extension]'}"


def find_scoring_key(reference: dict[str, Any], prefix: str) -> str | None:
    matches = sorted(
        key for key in reference if key == prefix or key.startswith(f"{prefix}_")
    )
    return matches[0] if matches else None


def input_file_record(
    workflow_id: str,
    workflow_dir: Path,
    relative_path: str,
    category: str,
    owner: str | None,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    resolved, path_error = resolve_reference(workflow_dir, relative_path)
    record: dict[str, Any] = {
        "workflow_id": workflow_id,
        "relative_path": relative_path,
        "category": category,
        "owner": owner,
        "extension": Path(relative_path).suffix.lower(),
        "exists": False,
        "size_bytes": None,
        "sha256": None,
        "integrity": "not_checked",
        "integrity_error": None,
    }
    if path_error:
        errors.append(f"{relative_path}: {path_error}")
        record["integrity_error"] = path_error
        return record, errors
    assert resolved is not None
    if not resolved.is_file():
        errors.append(f"missing referenced input file: {relative_path}")
        record["integrity"] = "failed"
        record["integrity_error"] = "file does not exist"
        return record, errors

    record["exists"] = True
    record["size_bytes"] = resolved.stat().st_size
    record["sha256"] = sha256_file(resolved)
    status, integrity_error = validate_file(resolved)
    record["integrity"] = status
    record["integrity_error"] = integrity_error
    if status == "failed":
        errors.append(f"unreadable input file {relative_path}: {integrity_error}")
    return record, errors


def audit_workflow(
    workflow_dir: Path,
    metadata_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    task_path = workflow_dir / TASK_FILENAME
    if not task_path.is_file():
        return {
            "workflow_id": workflow_dir.name,
            "folder": workflow_dir.name,
            "status": "blocked",
            "errors": [f"missing {TASK_FILENAME}"],
            "warnings": [],
            "files": [],
        }

    try:
        task = load_json(task_path)
    except (OSError, json.JSONDecodeError) as error:
        return {
            "workflow_id": workflow_dir.name,
            "folder": workflow_dir.name,
            "status": "blocked",
            "errors": [f"cannot parse {TASK_FILENAME}: {error}"],
            "warnings": [],
            "files": [],
        }

    if not isinstance(task, dict):
        errors.append("task description must be a JSON object")
        task = {}

    workflow_id = str(task.get("id", workflow_dir.name))
    if workflow_id != workflow_dir.name:
        errors.append(
            f"task id {workflow_id!r} does not match folder {workflow_dir.name!r}"
        )

    required_fields = {
        "id",
        "context",
        "required_amount",
        "currency",
        "source_files",
        "reference_outputs",
    }
    missing_fields = sorted(required_fields - set(task))
    if missing_fields:
        errors.append(f"missing task fields: {', '.join(missing_fields)}")

    context = task.get("context")
    if not isinstance(context, str) or not context.strip():
        errors.append("context must be a non-empty string")

    claimed_amount = decimal_value(task.get("required_amount"))
    if claimed_amount is None:
        errors.append("required_amount is not a valid decimal")
    elif claimed_amount < 0:
        errors.append("required_amount is negative")

    currency = task.get("currency")
    if not isinstance(currency, str) or not currency.strip():
        errors.append("currency must be a non-empty string")
        currency = None

    source_files = task.get("source_files")
    if not isinstance(source_files, dict):
        errors.append("source_files must be an object")
        source_files = {}

    background = source_files.get("Background_information", [])
    if not isinstance(background, list) or any(
        not isinstance(item, str) for item in background
    ):
        errors.append("Background_information must be a list of filenames")
        background = []

    receipts = source_files.get("Receipts", {})
    if not isinstance(receipts, dict):
        errors.append("Receipts must be an object")
        receipts = {}

    file_specs: list[tuple[str, str, str | None]] = [
        (filename, "background", None) for filename in background
    ]
    receipt_owner_count = 0
    receipt_value_types: Counter[str] = Counter()
    for owner, value in receipts.items():
        if owner == "standard":
            continue
        receipt_owner_count += 1
        receipt_value_types[
            "string"
            if isinstance(value, str)
            else "list"
            if isinstance(value, list)
            else "other"
        ] += 1
        try:
            filenames = normalize_receipt_value(value)
        except TypeError as error:
            errors.append(f"invalid receipt mapping for {owner}: {error}")
            continue
        file_specs.extend((filename, "receipt", str(owner)) for filename in filenames)

    referenced_paths = [spec[0] for spec in file_specs]
    duplicate_references = sorted(
        path for path, count in Counter(referenced_paths).items() if count > 1
    )
    if duplicate_references:
        warnings.append(
            f"input files referenced more than once: {', '.join(duplicate_references)}"
        )

    file_records: list[dict[str, Any]] = []
    for relative_path, category, owner in file_specs:
        record, record_errors = input_file_record(
            workflow_id, workflow_dir, relative_path, category, owner
        )
        file_records.append(record)
        errors.extend(record_errors)
        if record["extension"] not in SUPPORTED_INPUT_EXTENSIONS:
            warnings.append(f"unsupported input extension: {relative_path}")
        if record["integrity"] == "not_checked":
            warnings.append(f"integrity not checked: {relative_path}")

    reference_outputs = task.get("reference_outputs", [])
    if not isinstance(reference_outputs, list) or any(
        not isinstance(item, str) for item in reference_outputs
    ):
        errors.append("reference_outputs must be a filename list")
        reference_outputs = []

    leaked_answers = sorted(set(referenced_paths) & set(reference_outputs))
    if leaked_answers:
        errors.append(
            f"reference answers included in model inputs: {', '.join(leaked_answers)}"
        )

    reference_answer: dict[str, Any] | None = None
    reference_files: list[dict[str, Any]] = []
    for relative_path in reference_outputs:
        resolved, path_error = resolve_reference(workflow_dir, relative_path)
        reference_record = {
            "relative_path": relative_path,
            "exists": False,
            "sha256": None,
        }
        if path_error:
            errors.append(f"reference output {relative_path}: {path_error}")
        elif resolved is None or not resolved.is_file():
            errors.append(f"missing reference output: {relative_path}")
        else:
            reference_record["exists"] = True
            reference_record["sha256"] = sha256_file(resolved)
            if reference_answer is None and resolved.suffix.lower() == ".json":
                try:
                    loaded = load_json(resolved)
                    if isinstance(loaded, dict):
                        reference_answer = loaded
                    else:
                        errors.append(
                            f"reference answer {relative_path} is not a JSON object"
                        )
                except (OSError, json.JSONDecodeError) as error:
                    errors.append(
                        f"cannot parse reference answer {relative_path}: {error}"
                    )
        reference_files.append(reference_record)

    if reference_answer is None:
        errors.append("no usable JSON reference answer")
        reference_answer = {}

    scoring_points = {
        key: value
        for key, value in reference_answer.items()
        if key.startswith("Scoring_point_") and isinstance(value, dict)
    }
    if not scoring_points:
        errors.append("reference answer has no valid Scoring_point_* objects")

    score_sum = Decimal("0")
    invalid_score_keys: list[str] = []
    for key, value in scoring_points.items():
        score = decimal_value(value.get("score"))
        if score is None or score < 0:
            invalid_score_keys.append(key)
        else:
            score_sum += score
    if invalid_score_keys:
        errors.append(f"invalid scoring weights: {', '.join(invalid_score_keys)}")
    if scoring_points and score_sum != Decimal("1"):
        warnings.append(f"scoring weights sum to {score_sum}, not 1")

    flag_key = find_scoring_key(reference_answer, FLAG_PREFIX)
    amount_key = find_scoring_key(reference_answer, AMOUNT_PREFIX)
    flag = (
        normalize_flag(reference_answer.get(flag_key, {}).get("answer"))
        if flag_key
        else None
    )
    max_legal = (
        decimal_value(reference_answer.get(amount_key, {}).get("answer"))
        if amount_key
        else None
    )
    if flag_key is None or flag is None:
        errors.append("reference FLAG is missing or invalid")
    if amount_key is None or max_legal is None or max_legal < 0:
        errors.append("reference MAX_LEGAL is missing, invalid, or negative")

    metadata = metadata_by_id.get(workflow_id)
    if metadata is None:
        errors.append("workflow is missing from metadata.json")
        metadata = {}

    known_files = {TASK_FILENAME, *reference_outputs, *referenced_paths}
    extra_files = sorted(
        path.name
        for path in workflow_dir.iterdir()
        if path.is_file() and path.name not in known_files
    )
    if extra_files:
        warnings.append(
            f"unreferenced files in workflow folder: {', '.join(extra_files)}"
        )

    if errors:
        status = "blocked"
    elif warnings:
        status = "ready_with_warnings"
    else:
        status = "ready"

    return {
        "workflow_id": workflow_id,
        "folder": workflow_dir.name,
        "difficulty": metadata.get("difficulty"),
        "expected_types": metadata.get("expected_types", []),
        "claimed_amount": str(claimed_amount) if claimed_amount is not None else None,
        "currency": currency,
        "reference_flag": flag,
        "reference_max_legal": str(max_legal) if max_legal is not None else None,
        "reference_score_sum": str(score_sum),
        "scoring_point_count": len(scoring_points),
        "background_file_count": len(background),
        "receipt_owner_count": receipt_owner_count,
        "receipt_file_count": sum(
            1 for _, category, _ in file_specs if category == "receipt"
        ),
        "receipt_value_types": dict(sorted(receipt_value_types.items())),
        "input_file_count": len(file_records),
        "files": file_records,
        "reference_outputs": reference_files,
        "duplicate_references": duplicate_references,
        "extra_files": extra_files,
        "status": status,
        "errors": errors,
        "warnings": warnings,
    }


def git_commit(repo_dir: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def duplicate_hash_groups(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    locations_by_hash: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for workflow in manifest:
        for file_record in workflow.get("files", []):
            digest = file_record.get("sha256")
            if not digest:
                continue
            locations_by_hash[digest].append(
                {
                    "workflow_id": workflow["workflow_id"],
                    "relative_path": file_record["relative_path"],
                    "category": file_record["category"],
                }
            )

    groups = []
    for digest, locations in locations_by_hash.items():
        if len(locations) < 2:
            continue
        workflows = sorted({location["workflow_id"] for location in locations})
        groups.append(
            {
                "sha256": digest,
                "copy_count": len(locations),
                "workflow_count": len(workflows),
                "workflows": workflows,
                "locations": locations,
            }
        )
    return sorted(
        groups,
        key=lambda group: (
            -group["workflow_count"],
            -group["copy_count"],
            group["sha256"],
        ),
    )


def classify_duplicate_groups(groups: list[dict[str, Any]]) -> dict[str, int]:
    with_receipt = sum(
        1
        for group in groups
        if any(
            location.get("category") == "receipt"
            for location in group.get("locations", [])
        )
    )
    return {
        "with_receipt": with_receipt,
        "background_only": len(groups) - with_receipt,
    }


def amount_summary(manifest: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    amounts_by_currency: defaultdict[str, list[Decimal]] = defaultdict(list)
    for workflow in manifest:
        amount = decimal_value(workflow.get("claimed_amount"))
        currency = workflow.get("currency")
        if amount is not None and currency:
            amounts_by_currency[currency].append(amount)

    summaries: dict[str, dict[str, Any]] = {}
    for currency, amounts in sorted(amounts_by_currency.items()):
        summaries[currency] = {
            "count": len(amounts),
            "min": str(min(amounts)),
            "median": str(statistics.median(amounts)),
            "max": str(max(amounts)),
        }
    return summaries


def build_profile(
    workspace: Path,
    repo_dir: Path,
    manifest: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
    duplicate_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    status_counts = Counter(item["status"] for item in manifest)
    difficulty_counts = Counter(
        item.get("difficulty") or "missing" for item in manifest
    )
    currency_counts = Counter(item.get("currency") or "missing" for item in manifest)
    flag_counts = Counter(str(item.get("reference_flag")) for item in manifest)
    logic_counts = Counter(
        logic_type for item in manifest for logic_type in item.get("expected_types", [])
    )
    extension_counts = Counter(
        file_record["extension"] or "[none]"
        for item in manifest
        for file_record in item.get("files", [])
    )
    missing_inputs = sum(
        1
        for item in manifest
        for file_record in item.get("files", [])
        if not file_record.get("exists")
    )
    failed_integrity = sum(
        1
        for item in manifest
        for file_record in item.get("files", [])
        if file_record.get("integrity") == "failed"
    )
    answer_leaks = sum(
        1
        for item in manifest
        for error in item.get("errors", [])
        if error.startswith("reference answers included")
    )
    negative_claims = sum(
        1
        for item in manifest
        if (decimal_value(item.get("claimed_amount")) or Decimal("0")) < 0
    )
    max_above_claim = [
        item["workflow_id"]
        for item in manifest
        if decimal_value(item.get("reference_max_legal")) is not None
        and decimal_value(item.get("claimed_amount")) is not None
        and decimal_value(item["reference_max_legal"])
        > decimal_value(item["claimed_amount"])
    ]
    input_file_count = sum(item.get("input_file_count", 0) for item in manifest)
    input_bytes = sum(
        file_record.get("size_bytes") or 0
        for item in manifest
        for file_record in item.get("files", [])
    )
    repo_files = [
        path
        for path in repo_dir.rglob("*")
        if path.is_file() and ".git" not in path.parts
    ]
    repo_bytes = sum(path.stat().st_size for path in repo_files)
    cross_workflow_duplicates = [
        group for group in duplicate_groups if group["workflow_count"] > 1
    ]
    duplicate_classes = classify_duplicate_groups(cross_workflow_duplicates)

    benchmark_status = (
        "blocked"
        if status_counts["blocked"]
        else (
            "ready_with_warnings" if status_counts["ready_with_warnings"] else "ready"
        )
    )
    return {
        "schema_version": 1,
        "raw_files": [
            {
                "path": str(repo_dir.relative_to(workspace)).replace(os.sep, "/"),
                "git_commit": git_commit(repo_dir),
                "workflow_count": len(manifest),
                "file_count_excluding_git": len(repo_files),
                "bytes_excluding_git": repo_bytes,
                "raw_data_modified": False,
            }
        ],
        "attachment_mapping": [
            {
                "source": "task_description.json/source_files/Background_information",
                "target": "model_input/background",
            },
            {
                "source": "task_description.json/source_files/Receipts",
                "target": "model_input/receipt_with_owner",
            },
            {
                "source": "task_description.json/reference_outputs",
                "target": "labels_private/evaluation_only",
            },
            {
                "source": "metadata.json",
                "target": "split_and_analysis_metadata/not_model_input",
            },
        ],
        "fields": [
            {"name": "id", "source": TASK_FILENAME, "role": "workflow identifier"},
            {
                "name": "context",
                "source": TASK_FILENAME,
                "role": "application description",
            },
            {
                "name": "required_amount",
                "source": TASK_FILENAME,
                "role": "claimed amount",
            },
            {"name": "currency", "source": TASK_FILENAME, "role": "claim currency"},
            {
                "name": "source_files",
                "source": TASK_FILENAME,
                "role": "input attachment mapping",
            },
            {
                "name": "Scoring_point_*",
                "source": "reference_answer.json",
                "role": "private target",
            },
            {
                "name": "difficulty",
                "source": METADATA_FILENAME,
                "role": "stratification metadata",
            },
            {
                "name": "expected_types",
                "source": METADATA_FILENAME,
                "role": "logic-point metadata",
            },
        ],
        "quality": {
            "missingness": {
                "missing_referenced_input_files": missing_inputs,
                "workflows_without_usable_reference_answer": sum(
                    1
                    for item in manifest
                    if "no usable JSON reference answer" in item.get("errors", [])
                ),
                "workflows_missing_metadata": sum(
                    1
                    for item in manifest
                    if "workflow is missing from metadata.json"
                    in item.get("errors", [])
                ),
            },
            "duplicates": {
                "exact_input_file_hash_group_count": len(duplicate_groups),
                "cross_workflow_exact_hash_group_count": len(cross_workflow_duplicates),
                "cross_workflow_groups_with_receipts": duplicate_classes[
                    "with_receipt"
                ],
                "cross_workflow_background_only_groups": duplicate_classes[
                    "background_only"
                ],
                "groups": duplicate_groups,
            },
            "impossible_values": {
                "negative_claimed_amounts": negative_claims,
                "invalid_or_missing_flags": flag_counts.get("None", 0),
                "reference_answer_input_leaks": answer_leaks,
                "failed_file_integrity_checks": failed_integrity,
            },
            "outliers": {
                "max_legal_above_claim_count": len(max_above_claim),
                "max_legal_above_claim_workflows": max_above_claim,
                "note": "MAX_LEGAL above the claim is allowed by the benchmark and is not automatically an error.",
            },
        },
        "coverage": {
            "rows": len(manifest),
            "effective_sample_size": None,
            "effective_sample_size_note": "Source-family grouping is required before an independent training split can be claimed.",
            "input_file_references": input_file_count,
            "input_bytes": input_bytes,
            "time_range": None,
            "time_gaps": None,
        },
        "distribution_risks": {
            "class_imbalance": {
                "difficulty": dict(sorted(difficulty_counts.items())),
                "reference_flag": dict(sorted(flag_counts.items())),
                "logic_point_activation": dict(logic_counts.most_common()),
            },
            "rare_categories": [
                {"logic_type": name, "workflow_count": count}
                for name, count in sorted(
                    logic_counts.items(), key=lambda item: (item[1], item[0])
                )
                if count < 10
            ],
            "high_cardinality": [
                {
                    "field": "workflow_id",
                    "unique": len({item["workflow_id"] for item in manifest}),
                }
            ],
            "redundancy_warnings": [
                f"{duplicate_classes['with_receipt']} cross-workflow duplicate groups include receipts and are strong source-family signals.",
                f"{duplicate_classes['background_only']} duplicate groups contain only background files; shared policy templates must not automatically collapse into one source family.",
            ],
            "concentration_metrics": {
                "difficulty": dict(sorted(difficulty_counts.items())),
                "currency": dict(sorted(currency_counts.items())),
                "input_extension": dict(sorted(extension_counts.items())),
                "claimed_amount_by_currency": amount_summary(manifest),
                "metadata_rows": len(metadata_rows),
            },
        },
        "per_question_readiness": {
            "benchmark_evaluation": {
                "status": benchmark_status,
                "reason": "依据结构完整性、答案隔离和文件检查结果判断。",
            },
            "training_split": {
                "status": "blocked",
                "reason": "训练、验证和测试划分前，必须先归并同源材料及其语义编辑变体。",
            },
            "training_labels": {
                "status": "blocked",
                "reason": "参考答案保存整单判定与评分目标；条件级训练标签经原材料、政策和引用核验后批准。",
            },
            "document_parsing": {
                "status": "ready_with_warnings",
                "reason": "已完成压缩包、图片和PDF完整性检查；解析器负责版式、OCR、公式及嵌入媒体的内容核验。",
            },
        },
        "cleaned_files": [],
        "unresolved_risks": [
            "精确哈希识别字节一致的复用；编辑票据和同源事件材料结合事件信息审核并建立来源组。",
            "结构检查核验文件与字段完整性；政策解释和参考答案说明通过业务语义审核。",
            "DOCX/XLSX压缩包检查核验文件容器；解析检查逐项核验版式、公式、图片、页眉和合并单元格。",
            "PDF签名检查核验文件格式；页面读取和文字核验检查OCR质量及视觉可读性。",
        ],
        "workflow_status_counts": dict(sorted(status_counts.items())),
    }


def report_lines(profile: dict[str, Any], manifest: list[dict[str, Any]]) -> list[str]:
    raw = profile["raw_files"][0]
    quality = profile["quality"]
    distribution = profile["distribution_risks"]["concentration_metrics"]
    statuses = profile["workflow_status_counts"]
    blocked = [item for item in manifest if item["status"] == "blocked"]
    warnings = [item for item in manifest if item["status"] == "ready_with_warnings"]
    duplicate_summary = quality["duplicates"]

    lines = [
        "# ReimburseBench 数据完整性报告",
        "",
        "## 结论",
        "",
        f"已扫描 {raw['workflow_count']} 个工作流和 {profile['coverage']['input_file_references']} 个输入文件引用。",
        f"工作流状态：{', '.join(f'{key}={value}' for key, value in statuses.items())}。",
        "原始仓库以只读方式扫描，检查结果独立保存。",
        "",
        "数据处理依次完成来源组归并、固定划分、材料解析和标签核验，训练使用对应划分中的已批准监督。",
        "",
        "## 原始数据版本",
        "",
        f"- 路径：`{raw['path']}`",
        f"- Git提交：`{raw['git_commit']}`",
        f"- 仓库文件数：{raw['file_count_excluding_git']}",
        f"- 仓库大小：{raw['bytes_excluding_git']} 字节",
        "",
        "## 完整性检查",
        "",
        f"- 缺失的输入附件：{quality['missingness']['missing_referenced_input_files']}",
        f"- 不可用的参考答案：{quality['missingness']['workflows_without_usable_reference_answer']}",
        f"- 缺失metadata记录：{quality['missingness']['workflows_missing_metadata']}",
        f"- 文件完整性失败：{quality['impossible_values']['failed_file_integrity_checks']}",
        f"- 参考答案混入模型输入：{quality['impossible_values']['reference_answer_input_leaks']}",
        f"- 精确重复文件哈希组：{duplicate_summary['exact_input_file_hash_group_count']}",
        f"- 跨工作流重复哈希组：{duplicate_summary['cross_workflow_exact_hash_group_count']}",
        f"- 其中包含票据的重复组：{duplicate_summary['cross_workflow_groups_with_receipts']}",
        f"- 仅包含背景材料的重复组：{duplicate_summary['cross_workflow_background_only_groups']}",
        "",
        "## 数据分布",
        "",
        f"- 难度：{json.dumps(distribution['difficulty'], ensure_ascii=False, sort_keys=True)}",
        f"- 币种：{json.dumps(distribution['currency'], ensure_ascii=False, sort_keys=True)}",
        f"- 输入附件格式：{json.dumps(distribution['input_extension'], ensure_ascii=False, sort_keys=True)}",
        f"- FLAG：{json.dumps(profile['distribution_risks']['class_imbalance']['reference_flag'], ensure_ascii=False, sort_keys=True)}",
        "",
        "### 逻辑问题出现次数",
        "",
        "| 类型 | 工作流数 |",
        "|---|---:|",
    ]
    for name, count in profile["distribution_risks"]["class_imbalance"][
        "logic_point_activation"
    ].items():
        lines.append(f"| {name} | {count} |")

    lines.extend(
        [
            "",
            "## 需要人工继续确认的风险",
            "",
        ]
    )
    for risk in profile["unresolved_risks"]:
        lines.append(f"- {risk}")

    lines.extend(["", "## 工作流异常", ""])
    if not blocked and not warnings:
        lines.append("没有发现结构性错误或文件级警告。")
    else:
        for item in blocked:
            lines.append(
                f"- `{item['workflow_id']}` blocked：{'；'.join(item['errors'])}"
            )
        for item in warnings:
            lines.append(
                f"- `{item['workflow_id']}` warning：{'；'.join(item['warnings'])}"
            )

    lines.extend(
        [
            "",
            "## 当前可用性判断",
            "",
            "| 环节 | 状态 | 原因 |",
            "|---|---|---|",
        ]
    )
    for name, readiness in profile["per_question_readiness"].items():
        lines.append(f"| {name} | {readiness['status']} | {readiness['reason']} |")

    lines.extend(
        [
            "",
            "## 下一步",
            "",
            "1. 根据跨工作流重复文件和人工确认的同源材料建立 `source_family_id`。",
            "2. 按来源组划分训练、验证和测试集，同票据及其变体继承同一划分。",
            "3. 选择20个训练工作流做PDF、Word、Excel和图片解析试点。",
            "4. 将参考答案转换为候选条件标签，再进行程序和人工审核。",
            "",
            "详细到工作流和附件级的记录见 `workflow_manifest.jsonl`，机器可读画像见 `data_profile.json`。",
        ]
    )
    return lines


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=str)
        stream.write("\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def run_audit(workspace: Path, repo_dir: Path, output_dir: Path) -> dict[str, Any]:
    workflow_root = repo_dir / "workflow"
    metadata_path = repo_dir / METADATA_FILENAME
    if not workflow_root.is_dir():
        raise FileNotFoundError(f"workflow directory not found: {workflow_root}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata file not found: {metadata_path}")

    metadata_rows = load_json(metadata_path)
    if not isinstance(metadata_rows, list):
        raise ValueError("metadata.json must contain a list")
    metadata_by_id: dict[str, dict[str, Any]] = {}
    duplicate_metadata_ids: list[str] = []
    for row in metadata_rows:
        if not isinstance(row, dict) or "id" not in row:
            continue
        workflow_id = str(row["id"])
        if workflow_id in metadata_by_id:
            duplicate_metadata_ids.append(workflow_id)
        metadata_by_id[workflow_id] = row
    if duplicate_metadata_ids:
        raise ValueError(
            f"duplicate metadata IDs: {sorted(set(duplicate_metadata_ids))}"
        )

    workflow_dirs = sorted(
        (path for path in workflow_root.iterdir() if path.is_dir()),
        key=lambda path: [
            int(part) if part.isdigit() else part for part in path.name.split(".")
        ],
    )
    manifest = [audit_workflow(path, metadata_by_id) for path in workflow_dirs]
    duplicates = duplicate_hash_groups(manifest)
    profile = build_profile(workspace, repo_dir, manifest, metadata_rows, duplicates)

    write_jsonl(output_dir / "workflow_manifest.jsonl", manifest)
    write_json(output_dir / "data_profile.json", profile)
    report_path = output_dir / "data_report.md"
    report_path.write_text(
        "\n".join(report_lines(profile, manifest)) + "\n", encoding="utf-8"
    )
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    repo_dir = args.repo.resolve()
    output_dir = args.output.resolve()
    profile = run_audit(workspace, repo_dir, output_dir)
    print(
        json.dumps(
            {"workflow_status_counts": profile["workflow_status_counts"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
