"""Validate and integrate human-readable multimodal evidence observations.

This stage deliberately keeps visual readings separate from training labels.  It
adds traceable text/observations to normalized evidence and leaves every reading
in a pending-human-review state.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

READING_VERSION = "visual_reading_v1"
READING_STATUSES = {"clear", "partial", "needs_review"}
CONFIDENCE_LEVELS = {"high", "medium", "low"}
REVIEW_STATUSES = {"pending_human_review", "approved"}


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


def _require_string(reading: dict[str, Any], field: str) -> None:
    if not isinstance(reading.get(field), str) or not reading[field].strip():
        raise ValueError(f"{field} must be a non-empty string")


def validate_reading(reading: dict[str, Any]) -> None:
    """Validate one reading without coercing evidence values."""
    if reading.get("reading_version") != READING_VERSION:
        raise ValueError(
            f"unsupported reading_version: {reading.get('reading_version')!r}"
        )
    for field in ("workflow_id", "evidence_id", "raw_ref", "document_kind"):
        _require_string(reading, field)
    if reading.get("reading_status") not in READING_STATUSES:
        raise ValueError(f"invalid reading_status: {reading.get('reading_status')!r}")
    if reading.get("review_status") not in REVIEW_STATUSES:
        raise ValueError(f"invalid review_status: {reading.get('review_status')!r}")

    transcription = reading.get("transcription")
    if transcription is not None and not isinstance(transcription, str):
        raise ValueError("transcription must be a string or null")

    facts = reading.get("facts")
    if not isinstance(facts, list):
        raise ValueError("facts must be a list")
    for index, fact in enumerate(facts, start=1):
        if not isinstance(fact, dict):
            raise ValueError(f"fact {index} must be an object")
        if not isinstance(fact.get("field"), str) or not fact["field"].strip():
            raise ValueError(f"fact {index} field must be a non-empty string")
        if not isinstance(fact.get("value"), str):
            raise ValueError("fact value must be a string")
        for optional_field in (
            "normalized",
            "unit",
            "currency",
            "source_text",
            "notes",
        ):
            value = fact.get(optional_field)
            if value is not None and not isinstance(value, str):
                raise ValueError(
                    f"fact {index} {optional_field} must be a string or null"
                )
        if fact.get("confidence") not in CONFIDENCE_LEVELS:
            raise ValueError(
                f"fact {index} has invalid confidence: {fact.get('confidence')!r}"
            )

    for field in ("visual_observations", "uncertainties"):
        values = reading.get(field)
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value.strip() for value in values
        ):
            raise ValueError(f"{field} must be a list of non-empty strings")


def _reading_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record.get("workflow_id")), str(record.get("evidence_id"))


def validate_queue_coverage(
    queue: list[dict[str, Any]], readings: list[dict[str, Any]]
) -> None:
    """Require exactly one reading for every multimodal queue row."""
    expected_rows = [
        row for row in queue if "multimodal_reading" in row.get("actions", [])
    ]
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in expected_rows:
        key = _reading_key(row)
        if key in expected:
            raise ValueError(f"duplicate review queue key: {key}")
        expected[key] = row

    actual: dict[tuple[str, str], dict[str, Any]] = {}
    for reading in readings:
        validate_reading(reading)
        key = _reading_key(reading)
        if key in actual:
            raise ValueError(f"duplicate visual reading: {key}")
        actual[key] = reading

    missing = sorted(set(expected) - set(actual))
    if missing:
        raise ValueError(f"missing visual readings: {missing}")
    extra = sorted(set(actual) - set(expected))
    if extra:
        raise ValueError(f"unexpected visual readings: {extra}")
    for key, reading in actual.items():
        if reading["raw_ref"] != expected[key].get("raw_ref"):
            raise ValueError(f"raw_ref mismatch for visual reading {key}")


def _append_segment(
    evidence: dict[str, Any],
    *,
    kind: str,
    text: str,
    reading: dict[str, Any],
    note: str | None = None,
) -> None:
    locator: dict[str, Any] = {"scope": "whole_image"}
    if evidence.get("views"):
        locator["view_id"] = evidence["views"][0].get("view_id")
    segment = {
        "segment_id": f"{evidence['evidence_id']}:t{len(evidence['segments']) + 1:03d}",
        "kind": kind,
        "text": text,
        "source_locator": locator,
        "reading_version": reading["reading_version"],
    }
    if note is not None:
        segment["note"] = note
    evidence["segments"].append(segment)


def apply_visual_readings(
    bundles: list[dict[str, Any]], readings: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return a deep-copied normalized bundle with visual readings attached."""
    merged = copy.deepcopy(bundles)
    evidence_index: dict[tuple[str, str], dict[str, Any]] = {}
    for bundle in merged:
        workflow_id = str(bundle.get("workflow_id"))
        for evidence in bundle.get("evidence", []):
            key = (workflow_id, str(evidence.get("evidence_id")))
            if key in evidence_index:
                raise ValueError(f"duplicate normalized evidence key: {key}")
            evidence_index[key] = evidence

    for reading in readings:
        validate_reading(reading)
        key = _reading_key(reading)
        evidence = evidence_index.get(key)
        if evidence is None:
            raise ValueError(f"visual reading target not found: {key}")
        if evidence.get("type") != "image":
            raise ValueError(f"visual reading target is not an image: {key}")
        if evidence.get("raw_ref") != reading["raw_ref"]:
            raise ValueError(f"raw_ref mismatch for normalized evidence {key}")

        evidence.setdefault("segments", [])
        transcription = reading.get("transcription")
        if transcription is not None and transcription.strip():
            _append_segment(
                evidence,
                kind="multimodal_text",
                text=transcription.strip(),
                reading=reading,
            )
            evidence["text"] = transcription.strip()
        for observation in reading["visual_observations"]:
            _append_segment(
                evidence,
                kind="visual_observation",
                text="",
                reading=reading,
                note=observation,
            )

        evidence.setdefault("metadata", {})["visual_reading"] = copy.deepcopy(reading)
        evidence["parse_status"] = "multimodal_read_pending_review"
        warnings = [
            warning
            for warning in evidence.get("warnings", [])
            if warning != "multimodal_reading_required"
        ]
        if "human_review_required" not in warnings:
            warnings.append("human_review_required")
        if reading["uncertainties"] and "visual_reading_uncertainty" not in warnings:
            warnings.append("visual_reading_uncertainty")
        evidence["warnings"] = warnings

    return merged


def build_summary(readings: list[dict[str, Any]]) -> dict[str, Any]:
    fact_count = sum(len(reading["facts"]) for reading in readings)
    uncertainty_count = sum(len(reading["uncertainties"]) for reading in readings)
    unresolved_currency_count = sum(
        1
        for reading in readings
        for fact in reading["facts"]
        if "amount" in fact["field"] and fact.get("currency") is None
    )
    return {
        "reading_version": READING_VERSION,
        "reading_count": len(readings),
        "fact_count": fact_count,
        "uncertainty_count": uncertainty_count,
        "unresolved_currency_fact_count": unresolved_currency_count,
        "reading_status_counts": dict(
            sorted(Counter(row["reading_status"] for row in readings).items())
        ),
        "document_kind_counts": dict(
            sorted(Counter(row["document_kind"] for row in readings).items())
        ),
        "review_status_counts": dict(
            sorted(Counter(row["review_status"] for row in readings).items())
        ),
        "training_label_status": "not_generated",
    }


def build_review_queue(readings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "workflow_id": reading["workflow_id"],
            "evidence_id": reading["evidence_id"],
            "raw_ref": reading["raw_ref"],
            "document_kind": reading["document_kind"],
            "reading_status": reading["reading_status"],
            "uncertainties": reading["uncertainties"],
            "review_scope": "key_values_and_visual_semantics",
            "review_status": reading["review_status"],
        }
        for reading in readings
        if reading["review_status"] == "pending_human_review"
    ]


def render_report(summary: dict[str, Any]) -> str:
    status_lines = "\n".join(
        f"- `{name}`: {count}"
        for name, count in summary["reading_status_counts"].items()
    )
    return f"""# Pilot 20 视觉读取报告

## 本阶段做了什么

- 对标准化队列中的 {summary["reading_count"]} 张图片进行了关键字段读取。
- 共记录 {summary["fact_count"]} 个结构化事实和 {summary["uncertainty_count"]} 条不确定性。
- 读取结果合入新输入文件，原始附件与标准化输入分别保留。
- 读取结果进入事实核验队列，训练标签由后续条件级审核流程生成。

## 读取状态

{status_lines}

## 事实与来源记录

- 有 {summary["unresolved_currency_fact_count"]} 个金额事实保留币种未决标记，按来源补充核验。
- `visual_observation` 保存原图可见内容及观察说明，文字引用单独绑定原文片段。
- `pending_human_review` 将金额、日期、人员和语义关系送入事实审核；批准结果进入训练监督。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-dir", type=Path, default=Path("data/normalized"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalized_dir = args.normalized_dir
    queue_path = normalized_dir / "pilot20_review_queue.jsonl"
    readings_path = normalized_dir / "pilot20_visual_readings.jsonl"
    inputs_path = normalized_dir / "pilot20_inputs.jsonl"

    queue = load_jsonl(queue_path)
    readings = load_jsonl(readings_path)
    bundles = load_jsonl(inputs_path)
    validate_queue_coverage(queue, readings)
    merged = apply_visual_readings(bundles, readings)
    summary = build_summary(readings)
    review_queue = build_review_queue(readings)

    merged_path = normalized_dir / "pilot20_inputs_with_visual.jsonl"
    review_path = normalized_dir / "pilot20_visual_review_queue.jsonl"
    summary_path = normalized_dir / "pilot20_visual_summary.json"
    report_path = normalized_dir / "pilot20_visual_report.md"
    manifest_path = normalized_dir / "pilot20_visual_stage_manifest.json"

    write_jsonl(merged_path, merged)
    write_jsonl(review_path, review_queue)
    atomic_write_text(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(report_path, render_report(summary))

    manifest = {
        "stage": "pilot20_visual_reading",
        "reading_version": READING_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            path.name: {"sha256": sha256_file(path)}
            for path in (queue_path, readings_path, inputs_path)
        },
        "outputs": {
            path.name: {"sha256": sha256_file(path)}
            for path in (merged_path, review_path, summary_path, report_path)
        },
        "risk_level": "high",
        "requires_human_review": True,
        "training_labels_generated": False,
    }
    atomic_write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


if __name__ == "__main__":
    main()
