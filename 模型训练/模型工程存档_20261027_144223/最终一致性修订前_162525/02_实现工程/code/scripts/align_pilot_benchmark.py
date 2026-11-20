"""Align normalized pilot inputs with train-only benchmark reference targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

ALIGNMENT_VERSION = "benchmark_alignment_v1"
PAIR_VERSION = "pilot20_benchmark_pair_v1"
NUMBER_PATTERN = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.])")
FORBIDDEN_INPUT_KEYS = {
    "expected_answer",
    "reference_answer",
    "reference_flag",
    "reference_max_legal",
    "scoring_points",
    "target",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            records.append(value)
    return records


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8", newline="\n")
    temporary_path.replace(path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    atomic_write_text(path, content)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index_by_workflow(
    records: list[dict[str, Any]], record_name: str
) -> dict[str, dict[str, Any]]:
    index = {}
    for record in records:
        workflow_id = record.get("workflow_id")
        if not isinstance(workflow_id, str) or not workflow_id:
            raise ValueError(f"{record_name} has no workflow_id")
        if workflow_id in index:
            raise ValueError(f"duplicate {record_name} workflow_id: {workflow_id}")
        index[workflow_id] = record
    return index


def validate_pair_coverage(
    bundles: list[dict[str, Any]], targets: list[dict[str, Any]]
) -> None:
    bundle_index = _index_by_workflow(bundles, "pilot bundle")
    target_index = _index_by_workflow(targets, "benchmark target")
    missing = sorted(set(bundle_index) - set(target_index))
    if missing:
        raise ValueError(f"missing benchmark targets: {missing}")
    extra = sorted(set(target_index) - set(bundle_index))
    if extra:
        raise ValueError(f"unexpected benchmark targets: {extra}")
    for workflow_id, bundle in bundle_index.items():
        target = target_index[workflow_id]
        if target.get("split") != "train":
            raise ValueError(f"non-train target supplied for {workflow_id}")
        if bundle.get("source_family_id") != target.get("source_family_id"):
            raise ValueError(f"source_family_id mismatch for {workflow_id}")


def _decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{label} must be a decimal-compatible value")
    try:
        return Decimal(str(value).replace(",", ""))
    except InvalidOperation as error:
        raise ValueError(f"invalid {label}: {value!r}") from error


def check_flag_consistency(
    bundle: dict[str, Any], target: dict[str, Any]
) -> dict[str, Any]:
    claimed = _decimal(bundle.get("input", {}).get("claimed_amount"), "claimed_amount")
    maximum = _decimal(
        target.get("target", {}).get("reference_max_legal"),
        "reference_max_legal",
    )
    actual_flag = target.get("target", {}).get("reference_flag")
    if actual_flag not in {0, 1}:
        raise ValueError(f"invalid reference_flag: {actual_flag!r}")
    expected_flag = 1 if maximum >= claimed else 0
    return {
        "claimed_amount": format(claimed, "f"),
        "reference_max_legal": format(maximum, "f"),
        "reference_flag": actual_flag,
        "expected_flag": expected_flag,
        "consistent": actual_flag == expected_flag,
    }


def _canonical_number(value: str) -> str:
    number = Decimal(value.replace(",", ""))
    normalized = format(number, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def _number_tokens(text: str) -> list[tuple[str, str]]:
    tokens = []
    seen = set()
    for match in NUMBER_PATTERN.finditer(text):
        original = match.group(0).replace(",", "")
        canonical = _canonical_number(original)
        if canonical in seen:
            continue
        seen.add(canonical)
        tokens.append((original, canonical))
    return tokens


def _collect_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in _collect_strings(item)]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _collect_strings(item)]
    return []


def _evidence_corpus(bundle: dict[str, Any]) -> str:
    texts = _collect_strings(bundle.get("input", {}))
    for evidence in bundle.get("evidence", []):
        if isinstance(evidence.get("text"), str):
            texts.append(evidence["text"])
        for segment in evidence.get("segments", []):
            for field in ("text", "note"):
                if isinstance(segment.get(field), str):
                    texts.append(segment[field])
        visual_reading = evidence.get("metadata", {}).get("visual_reading")
        if isinstance(visual_reading, dict):
            texts.extend(_collect_strings(visual_reading))
    return "\n".join(texts)


def check_problem_numeric_traceability(
    bundle: dict[str, Any], target: dict[str, Any]
) -> dict[str, Any]:
    reports = target.get("target", {}).get("problem_reports", [])
    report_text = "\n".join(str(report) for report in reports)
    report_tokens = _number_tokens(report_text)
    evidence_numbers = {
        canonical for _, canonical in _number_tokens(_evidence_corpus(bundle))
    }
    matched = [
        original
        for original, canonical in report_tokens
        if canonical in evidence_numbers
    ]
    unmatched = [
        original
        for original, canonical in report_tokens
        if canonical not in evidence_numbers
    ]
    if not report_tokens:
        status = "not_applicable"
    elif not unmatched:
        status = "full"
    elif matched:
        status = "partial"
    else:
        status = "none"
    return {
        "status": status,
        "mentions": [original for original, _ in report_tokens],
        "matched": matched,
        "unmatched": unmatched,
        "interpretation": "lexical_numeric_check_only",
    }


def _find_forbidden_keys(value: Any, path: str = "input") -> list[str]:
    findings = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key).lower() in FORBIDDEN_INPUT_KEYS:
                findings.append(child_path)
            findings.extend(_find_forbidden_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_find_forbidden_keys(child, f"{path}[{index}]"))
    return findings


def _visual_reading_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    statuses = Counter()
    pending_review_count = 0
    uncertainty_count = 0
    for evidence in bundle.get("evidence", []):
        reading = evidence.get("metadata", {}).get("visual_reading")
        if not isinstance(reading, dict):
            continue
        statuses[str(reading.get("reading_status"))] += 1
        if reading.get("review_status") == "pending_human_review":
            pending_review_count += 1
        uncertainty_count += len(reading.get("uncertainties", []))
    return {
        "reading_count": sum(statuses.values()),
        "reading_status_counts": dict(sorted(statuses.items())),
        "pending_human_review_count": pending_review_count,
        "uncertainty_count": uncertainty_count,
    }


def build_alignment_record(
    bundle: dict[str, Any], target: dict[str, Any]
) -> dict[str, Any]:
    flag_check = check_flag_consistency(bundle, target)
    traceability = check_problem_numeric_traceability(bundle, target)
    forbidden_keys = _find_forbidden_keys(bundle)
    visual_summary = _visual_reading_summary(bundle)

    hold_reasons = []
    target_annotation = target.get("annotation", {})
    if target_annotation.get("status") == "needs_review":
        hold_reasons.append("benchmark_reference_needs_review")
    if not flag_check["consistent"]:
        hold_reasons.append("reference_flag_amount_mismatch")
    if forbidden_keys:
        hold_reasons.append("target_leakage_in_input")

    warnings = []
    if visual_summary["pending_human_review_count"]:
        warnings.append("visual_readings_pending_review")
    if traceability["status"] in {"partial", "none"}:
        warnings.append("problem_numbers_not_fully_traceable")
    if target_annotation.get("status") == "reference_only":
        warnings.append("benchmark_reference_not_component_gold_label")

    if hold_reasons:
        training_status = "hold"
    elif warnings:
        training_status = "ready_with_warnings"
    else:
        training_status = "ready"

    return {
        "schema_version": "1.0",
        "alignment_version": ALIGNMENT_VERSION,
        "workflow_id": bundle["workflow_id"],
        "source_family_id": bundle["source_family_id"],
        "input_version": bundle["input_version"],
        "target_version": target["target_version"],
        "split": target["split"],
        "checks": {
            "workflow_id_match": bundle["workflow_id"] == target["workflow_id"],
            "source_family_id_match": (
                bundle["source_family_id"] == target["source_family_id"]
            ),
            "train_split_only": target["split"] == "train",
            "target_leakage_detected": bool(forbidden_keys),
            "forbidden_input_key_paths": forbidden_keys,
            "flag_amount_consistency": flag_check,
            "problem_numeric_traceability": traceability,
        },
        "visual_reading_summary": visual_summary,
        "benchmark_annotation": target_annotation,
        "training_status": training_status,
        "hold_reasons": hold_reasons,
        "warnings": warnings,
        "allowed_label_scope": "benchmark_end_to_end_only",
    }


def build_benchmark_pair(
    bundle: dict[str, Any], target: dict[str, Any]
) -> dict[str, Any]:
    is_held = target.get("annotation", {}).get("status") == "needs_review"
    return {
        "schema_version": "1.0",
        "pair_version": PAIR_VERSION,
        "workflow_id": bundle["workflow_id"],
        "source_family_id": bundle["source_family_id"],
        "input_version": bundle["input_version"],
        "target_version": target["target_version"],
        "input": bundle["input"],
        "evidence": bundle["evidence"],
        "target": target["target"],
        "annotation": {
            "label_source": "benchmark_reference",
            "source_annotation_status": target.get("annotation", {}).get("status"),
            "use_for_end_to_end_supervision": not is_held,
            "approved_for_component_training": False,
            "notes": [
                "The benchmark target supervises the final audit output only.",
                "It does not approve condition, relation, citation, or reading labels.",
            ],
        },
    }


def build_summary(
    alignment_records: list[dict[str, Any]], pair_count: int
) -> dict[str, Any]:
    return {
        "alignment_version": ALIGNMENT_VERSION,
        "workflow_count": len(alignment_records),
        "benchmark_pair_count": pair_count,
        "training_status_counts": dict(
            sorted(Counter(row["training_status"] for row in alignment_records).items())
        ),
        "numeric_traceability_counts": dict(
            sorted(
                Counter(
                    row["checks"]["problem_numeric_traceability"]["status"]
                    for row in alignment_records
                ).items()
            )
        ),
        "reference_flag_counts": dict(
            sorted(
                Counter(
                    str(row["checks"]["flag_amount_consistency"]["reference_flag"])
                    for row in alignment_records
                ).items()
            )
        ),
        "target_leakage_count": sum(
            row["checks"]["target_leakage_detected"] for row in alignment_records
        ),
        "flag_amount_mismatch_count": sum(
            not row["checks"]["flag_amount_consistency"]["consistent"]
            for row in alignment_records
        ),
        "component_training_label_count": 0,
        "reading_sft_label_count": 0,
    }


def render_report(summary: dict[str, Any], hold_records: list[dict[str, Any]]) -> str:
    hold_lines = (
        "\n".join(
            f"- `{row['workflow_id']}`：{', '.join(row['hold_reasons'])}"
            for row in hold_records
        )
        or "- 无"
    )
    return f"""# Pilot 20 Benchmark 对齐报告

## 结果

- 已对齐 {summary["workflow_count"]} 个训练集试点工作流。
- 可生成端到端 Benchmark 监督配对：{summary["benchmark_pair_count"]} 个。
- 目标泄漏：{summary["target_leakage_count"]} 个。
- FLAG 与 `MAX_LEGAL >= CLAIMED` 规则不一致：{summary["flag_amount_mismatch_count"]} 个。
- 数字来源检查：{json.dumps(summary["numeric_traceability_counts"], ensure_ascii=False, sort_keys=True)}。

## 暂停样本

{hold_lines}

## 使用边界

- 配对标签来自公开 Benchmark 的最终答案，可用于端到端 Benchmark 监督实验。
- 这些答案不能自动拆成条件、关系、引用或三分金额标签。
- 当前没有生成三个新增神经部件的正式标签，也没有生成读取 SFT 金标。
- 验证集和测试集参考答案没有进入本输出。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        default=Path("data/normalized/pilot20_inputs_with_visual.jsonl"),
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("data/labels_private/train_reference_targets.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/training"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundles = load_jsonl(args.inputs)
    bundle_ids = {bundle["workflow_id"] for bundle in bundles}
    targets = [
        target
        for target in load_jsonl(args.targets)
        if target.get("workflow_id") in bundle_ids
    ]
    validate_pair_coverage(bundles, targets)
    target_index = _index_by_workflow(targets, "benchmark target")

    alignment_records = []
    benchmark_pairs = []
    hold_records = []
    for bundle in bundles:
        target = target_index[bundle["workflow_id"]]
        alignment = build_alignment_record(bundle, target)
        alignment_records.append(alignment)
        if alignment["training_status"] == "hold":
            hold_records.append(alignment)
        else:
            benchmark_pairs.append(build_benchmark_pair(bundle, target))

    summary = build_summary(alignment_records, len(benchmark_pairs))
    output_dir = args.output_dir
    alignment_path = output_dir / "pilot20_benchmark_alignment.jsonl"
    pairs_path = output_dir / "pilot20_benchmark_pairs.jsonl"
    hold_path = output_dir / "pilot20_benchmark_hold_queue.jsonl"
    summary_path = output_dir / "pilot20_benchmark_summary.json"
    report_path = output_dir / "pilot20_benchmark_report.md"
    manifest_path = output_dir / "pilot20_benchmark_stage_manifest.json"

    write_jsonl(alignment_path, alignment_records)
    write_jsonl(pairs_path, benchmark_pairs)
    write_jsonl(hold_path, hold_records)
    atomic_write_text(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(report_path, render_report(summary, hold_records))

    manifest = {
        "stage": "pilot20_benchmark_alignment",
        "alignment_version": ALIGNMENT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            path.as_posix(): {"sha256": sha256_file(path)}
            for path in (args.inputs, args.targets)
        },
        "outputs": {
            path.as_posix(): {"sha256": sha256_file(path)}
            for path in (
                alignment_path,
                pairs_path,
                hold_path,
                summary_path,
                report_path,
            )
        },
        "risk_level": "high",
        "rollback": "Remove data/training/pilot20_benchmark_* outputs.",
        "validation_and_test_targets_loaded": False,
        "component_training_labels_generated": False,
        "reading_sft_labels_generated": False,
    }
    atomic_write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


if __name__ == "__main__":
    main()
