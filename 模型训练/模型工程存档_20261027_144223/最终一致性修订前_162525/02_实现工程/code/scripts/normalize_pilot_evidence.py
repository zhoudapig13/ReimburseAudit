"""Normalize selected pilot attachments while preserving raw evidence references."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from datetime import date, datetime, time, timezone
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any

import pymupdf
from docx import Document
from docx.table import Table
from openpyxl import load_workbook
from PIL import Image

PARSER_VERSION = "parser_v1"
SUPPORTED_TYPES = {"docx", "image", "json", "pdf", "xlsx"}


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


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_bytes(content)
    temporary_path.replace(path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    atomic_write_text(path, content)


def normalize_text(value: str) -> str:
    lines = [" ".join(line.split()) for line in value.replace("\x00", "").splitlines()]
    return "\n".join(line for line in lines if line).strip()


def resolve_source_path(raw_root: Path, raw_ref: str) -> Path:
    candidate_ref = PurePosixPath(str(raw_ref).replace("\\", "/"))
    if candidate_ref.is_absolute() or ".." in candidate_ref.parts:
        raise ValueError(f"unsafe raw_ref: {raw_ref!r}")
    root = raw_root.resolve()
    candidate = (root / Path(*candidate_ref.parts)).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"unsafe raw_ref outside raw root: {raw_ref!r}")
    if not candidate.is_file():
        raise FileNotFoundError(f"raw evidence file not found: {raw_ref}")
    return candidate


def _base_record(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": source["evidence_id"],
        "raw_ref": source["raw_ref"],
        "type": source["type"],
        "role_hint": source.get("role_hint"),
        "owners": list(source.get("owners", [])),
        "parser_version": PARSER_VERSION,
        "text": None,
        "segments": [],
        "views": [],
        "parse_status": "parsed",
        "warnings": [],
        "metadata": {},
    }


def _segment(
    record: dict[str, Any], kind: str, text: str, locator: dict[str, Any], **extra: Any
) -> None:
    normalized = normalize_text(text)
    if not normalized and not extra:
        return
    index = len(record["segments"]) + 1
    value = {
        "segment_id": f"{record['evidence_id']}:t{index:03d}",
        "kind": kind,
        "text": normalized,
        "source_locator": locator,
    }
    value.update(extra)
    record["segments"].append(value)


def _set_combined_text(record: dict[str, Any]) -> None:
    texts = [segment["text"] for segment in record["segments"] if segment["text"]]
    record["text"] = "\n\n".join(texts) or None


def _parse_image(source: dict[str, Any], source_path: Path) -> dict[str, Any]:
    record = _base_record(source)
    with Image.open(source_path) as image:
        image.verify()
    with Image.open(source_path) as image:
        record["metadata"] = {
            "format": image.format,
            "mode": image.mode,
            "width_px": image.width,
            "height_px": image.height,
        }
    record["views"] = [
        {
            "view_id": f"{record['evidence_id']}:v001",
            "source_id": record["evidence_id"],
            "kind": "original_image",
            "raw_ref": record["raw_ref"],
        }
    ]
    record["parse_status"] = "visual_only"
    record["warnings"].append("multimodal_reading_required")
    return record


def _parse_pdf(
    source: dict[str, Any], source_path: Path, views_dir: Path
) -> dict[str, Any]:
    record = _base_record(source)
    textless_pages = []
    page_sizes = []
    document = pymupdf.open(source_path)
    try:
        for page_index, page in enumerate(document, start=1):
            page_sizes.append(
                {
                    "page_number": page_index,
                    "width_pt": round(page.rect.width, 3),
                    "height_pt": round(page.rect.height, 3),
                }
            )
            page_has_text = False
            blocks = page.get_text("blocks", sort=True)
            for block_index, block in enumerate(blocks, start=1):
                block_text = normalize_text(str(block[4]))
                if not block_text:
                    continue
                page_has_text = True
                _segment(
                    record,
                    "pdf_text_block",
                    block_text,
                    {
                        "page_number": page_index,
                        "block_index": block_index,
                        "bbox": [round(float(value), 3) for value in block[:4]],
                    },
                )
            if not page_has_text:
                textless_pages.append(page_index)
                target = views_dir / f"page-{page_index:04d}.png"
                pixmap = page.get_pixmap(dpi=150, alpha=False)
                atomic_write_bytes(target, pixmap.tobytes("png"))
                record["views"].append(
                    {
                        "view_id": f"{record['evidence_id']}:v{len(record['views']) + 1:03d}",
                        "source_id": record["evidence_id"],
                        "kind": "pdf_page_image",
                        "page_number": page_index,
                        "file_path": target.as_posix(),
                    }
                )
    finally:
        document.close()
    record["metadata"] = {
        "page_count": len(page_sizes),
        "page_sizes": page_sizes,
        "textless_pages": textless_pages,
    }
    _set_combined_text(record)
    if textless_pages:
        record["warnings"].append("multimodal_reading_required")
        record["parse_status"] = (
            "visual_only" if record["text"] is None else "parsed_with_warnings"
        )
    return record


def _table_row_text(headers: list[str], cells: list[str]) -> str:
    parts = []
    for index, value in enumerate(cells):
        if not value:
            continue
        header = headers[index] if index < len(headers) else ""
        parts.append(f"{header}: {value}" if header and header != value else value)
    return " | ".join(parts)


def _extract_docx_media(
    source_path: Path, record: dict[str, Any], views_dir: Path
) -> None:
    with zipfile.ZipFile(source_path) as archive:
        media_names = sorted(
            name for name in archive.namelist() if name.startswith("word/media/")
        )
        for index, media_name in enumerate(media_names, start=1):
            suffix = Path(media_name).suffix.lower() or ".bin"
            target = views_dir / f"embedded-{index:03d}{suffix}"
            atomic_write_bytes(target, archive.read(media_name))
            record["views"].append(
                {
                    "view_id": f"{record['evidence_id']}:v{len(record['views']) + 1:03d}",
                    "source_id": record["evidence_id"],
                    "kind": "docx_embedded_image",
                    "container_path": media_name,
                    "file_path": target.as_posix(),
                }
            )
    record["metadata"]["embedded_image_count"] = len(media_names)
    if media_names:
        record["warnings"].append("multimodal_reading_required")


def _parse_docx(
    source: dict[str, Any], source_path: Path, views_dir: Path
) -> dict[str, Any]:
    record = _base_record(source)
    document = Document(source_path)
    paragraph_count = 0
    table_count = 0
    for block_index, block in enumerate(document.iter_inner_content(), start=1):
        if isinstance(block, Table):
            table_count += 1
            rows = [
                [normalize_text(cell.text) for cell in row.cells] for row in block.rows
            ]
            headers = rows[0] if rows else []
            for row_index, cells in enumerate(rows, start=1):
                _segment(
                    record,
                    "docx_table_row",
                    _table_row_text(headers, cells),
                    {
                        "body_block": block_index,
                        "table_index": table_count,
                        "row_index": row_index,
                    },
                    headers=headers,
                    cells=cells,
                )
            continue
        text = normalize_text(block.text)
        if not text:
            continue
        paragraph_count += 1
        _segment(
            record,
            "docx_paragraph",
            text,
            {"body_block": block_index, "paragraph_index": paragraph_count},
            style=block.style.name if block.style is not None else None,
        )

    seen_parts = set()
    for section_index, section in enumerate(document.sections, start=1):
        for region_name, region in (
            ("header", section.header),
            ("footer", section.footer),
        ):
            part_name = str(region.part.partname)
            if part_name in seen_parts:
                continue
            seen_parts.add(part_name)
            for paragraph_index, paragraph in enumerate(region.paragraphs, start=1):
                text = normalize_text(paragraph.text)
                if text:
                    _segment(
                        record,
                        f"docx_{region_name}",
                        text,
                        {
                            "section_index": section_index,
                            "paragraph_index": paragraph_index,
                        },
                    )
    record["metadata"] = {
        "body_paragraph_count": paragraph_count,
        "table_count": table_count,
        "section_count": len(document.sections),
    }
    _extract_docx_media(source_path, record, views_dir)
    _set_combined_text(record)
    if record["warnings"]:
        record["parse_status"] = "parsed_with_warnings"
    return record


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, float):
        return format(value, ".15g")
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _cell_record(formula_cell: Any, cached_cell: Any) -> dict[str, Any]:
    value = formula_cell.value
    is_formula = formula_cell.data_type == "f"
    record = {
        "coordinate": formula_cell.coordinate,
        "value": _json_value(value),
        "value_type": "formula" if is_formula else formula_cell.data_type,
        "number_format": formula_cell.number_format,
    }
    if is_formula:
        record["formula"] = str(value)
        record["cached_value"] = _json_value(cached_cell.value)
    if formula_cell.comment is not None:
        record["note"] = normalize_text(formula_cell.comment.text)
    if formula_cell.hyperlink is not None:
        record["hyperlink"] = formula_cell.hyperlink.target
    return record


def _xlsx_row_text(sheet_title: str, cells: list[dict[str, Any]]) -> str:
    values = []
    for cell in cells:
        value = cell.get("formula", cell.get("value"))
        if value is not None:
            values.append(f"{cell['coordinate']}={value}")
    return f"Sheet {sheet_title}: " + " | ".join(values)


def _parse_xlsx(source: dict[str, Any], source_path: Path) -> dict[str, Any]:
    record = _base_record(source)
    formula_book = load_workbook(source_path, data_only=False, read_only=False)
    cached_book = load_workbook(source_path, data_only=True, read_only=False)
    unresolved_formula_count = 0
    sheet_metadata = []
    try:
        for sheet_index, sheet in enumerate(formula_book.worksheets, start=1):
            cached_sheet = cached_book[sheet.title]
            formula_count = 0
            note_count = 0
            nonempty_row_count = 0
            for row_index in range(1, sheet.max_row + 1):
                cells = []
                for column_index in range(1, sheet.max_column + 1):
                    formula_cell = sheet.cell(row=row_index, column=column_index)
                    if formula_cell.value is None and formula_cell.comment is None:
                        continue
                    cached_cell = cached_sheet.cell(row=row_index, column=column_index)
                    cell = _cell_record(formula_cell, cached_cell)
                    if formula_cell.data_type == "f":
                        formula_count += 1
                        if cached_cell.value is None:
                            unresolved_formula_count += 1
                    if formula_cell.comment is not None:
                        note_count += 1
                    cells.append(cell)
                if not cells:
                    continue
                nonempty_row_count += 1
                _segment(
                    record,
                    "xlsx_row",
                    _xlsx_row_text(sheet.title, cells),
                    {"sheet": sheet.title, "row": row_index},
                    cells=cells,
                )
            sheet_metadata.append(
                {
                    "sheet_index": sheet_index,
                    "title": sheet.title,
                    "state": sheet.sheet_state,
                    "max_row": sheet.max_row,
                    "max_column": sheet.max_column,
                    "nonempty_row_count": nonempty_row_count,
                    "formula_count": formula_count,
                    "note_count": note_count,
                    "merged_ranges": sorted(str(value) for value in sheet.merged_cells),
                }
            )
    finally:
        formula_book.close()
        cached_book.close()
    record["metadata"] = {
        "sheet_count": len(sheet_metadata),
        "sheets": sheet_metadata,
        "unresolved_formula_count": unresolved_formula_count,
    }
    _set_combined_text(record)
    if unresolved_formula_count:
        record["warnings"].append("formula_cache_missing")
        record["parse_status"] = "parsed_with_warnings"
    return record


def _parse_json(source: dict[str, Any], source_path: Path) -> dict[str, Any]:
    record = _base_record(source)
    with source_path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, parse_float=Decimal)
    if isinstance(value, dict):
        for key in sorted(value):
            field_value = _json_value(value[key])
            text = json.dumps(field_value, ensure_ascii=False, sort_keys=True)
            _segment(
                record,
                "json_field",
                f"{key}: {text}",
                {"json_pointer": f"/{key}"},
                field=key,
                value=field_value,
            )
    else:
        json_value = _json_value(value)
        _segment(
            record,
            "json_value",
            json.dumps(json_value, ensure_ascii=False, sort_keys=True),
            {"json_pointer": ""},
            value=json_value,
        )
    record["metadata"] = {"top_level_type": type(value).__name__}
    _set_combined_text(record)
    return record


def parse_evidence(
    source: dict[str, Any], *, raw_root: Path, views_root: Path, workflow_id: str
) -> dict[str, Any]:
    evidence_type = str(source.get("type", ""))
    if evidence_type not in SUPPORTED_TYPES:
        raise ValueError(f"unsupported evidence type: {evidence_type!r}")
    source_path = resolve_source_path(raw_root, str(source.get("raw_ref", "")))
    views_dir = views_root / workflow_id / str(source["evidence_id"])
    if evidence_type == "image":
        return _parse_image(source, source_path)
    if evidence_type == "pdf":
        return _parse_pdf(source, source_path, views_dir)
    if evidence_type == "docx":
        return _parse_docx(source, source_path, views_dir)
    if evidence_type == "xlsx":
        return _parse_xlsx(source, source_path)
    return _parse_json(source, source_path)


def normalize_bundle(
    bundle: dict[str, Any], *, raw_root: Path, views_root: Path
) -> dict[str, Any]:
    workflow_id = str(bundle.get("workflow_id", ""))
    if not workflow_id:
        raise ValueError("packed input is missing workflow_id")
    normalized_evidence = []
    for source in bundle.get("evidence", []):
        try:
            parsed = parse_evidence(
                source,
                raw_root=raw_root,
                views_root=views_root,
                workflow_id=workflow_id,
            )
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
            parsed = _base_record(source)
            parsed["parse_status"] = "failed"
            parsed["warnings"] = [f"parser_failure:{type(error).__name__}:{error}"]
        normalized_evidence.append(parsed)
    return {
        "schema_version": "1.0",
        "input_version": "normalized_bundle_v1",
        "parser_version": PARSER_VERSION,
        "workflow_id": workflow_id,
        "source_family_id": bundle.get("source_family_id"),
        "input": bundle.get("input", {}),
        "evidence": normalized_evidence,
    }


def build_review_queue(bundles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queue = []
    for bundle in bundles:
        for evidence in bundle["evidence"]:
            warnings = set(evidence["warnings"])
            actions = []
            if "multimodal_reading_required" in warnings:
                actions.append("multimodal_reading")
            if "formula_cache_missing" in warnings:
                actions.append("formula_cache_review")
            if evidence["parse_status"] == "failed":
                actions.append("parser_review")
            if not actions:
                continue
            queue.append(
                {
                    "workflow_id": bundle["workflow_id"],
                    "evidence_id": evidence["evidence_id"],
                    "raw_ref": evidence["raw_ref"],
                    "type": evidence["type"],
                    "actions": actions,
                    "view_ids": [view["view_id"] for view in evidence["views"]],
                    "warnings": evidence["warnings"],
                }
            )
    return queue


def build_summary(
    bundles: list[dict[str, Any]], queue: list[dict[str, Any]]
) -> dict[str, Any]:
    evidence = [item for bundle in bundles for item in bundle["evidence"]]
    return {
        "parser_version": PARSER_VERSION,
        "workflow_count": len(bundles),
        "evidence_count": len(evidence),
        "evidence_type_counts": dict(
            sorted(Counter(item["type"] for item in evidence).items())
        ),
        "parse_status_counts": dict(
            sorted(Counter(item["parse_status"] for item in evidence).items())
        ),
        "segment_count": sum(len(item["segments"]) for item in evidence),
        "view_count": sum(len(item["views"]) for item in evidence),
        "review_queue_count": len(queue),
        "review_action_counts": dict(
            sorted(
                Counter(action for row in queue for action in row["actions"]).items()
            )
        ),
        "warning_counts": dict(
            sorted(
                Counter(
                    warning for item in evidence for warning in item["warnings"]
                ).items()
            )
        ),
    }


def render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# 20 个训练工作流附件解析报告",
            "",
            "本阶段只做附件读取和结构化，不生成监督答案。图片、Word 内嵌图和扫描 PDF 页进入后续多模态读取队列。",
            "",
            f"- 工作流：{summary['workflow_count']}",
            f"- 附件：{summary['evidence_count']}",
            f"- 结构化片段：{summary['segment_count']}",
            f"- 生成视图：{summary['view_count']}",
            f"- 解析状态：{json.dumps(summary['parse_status_counts'], ensure_ascii=False)}",
            f"- 待处理附件：{summary['review_queue_count']}",
            f"- 待处理动作：{json.dumps(summary['review_action_counts'], ensure_ascii=False)}",
            "",
            "## 状态含义",
            "",
            "- `parsed`：现有本地解析器已取得结构化内容。",
            "- `parsed_with_warnings`：已取得部分内容，但仍有图片、扫描页或公式缓存问题。",
            "- `visual_only`：没有伪造 OCR 文本，必须后续读取原图或整页视图。",
            "- `failed`：解析器失败，必须人工或替代工具处理。",
            "",
        ]
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("data/pilot/pilot20_selection.jsonl"),
    )
    parser.add_argument(
        "--train-inputs", type=Path, default=Path("data/packed/train_inputs.jsonl")
    )
    parser.add_argument(
        "--raw-root", type=Path, default=Path("data_raw/ReimburseBench")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/normalized"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selection = load_jsonl(args.selection)
    bundle_by_id = {
        str(bundle["workflow_id"]): bundle for bundle in load_jsonl(args.train_inputs)
    }
    normalized = []
    for selected in sorted(selection, key=lambda row: row["selection_rank"]):
        workflow_id = str(selected["workflow_id"])
        bundle = bundle_by_id.get(workflow_id)
        if bundle is None:
            raise ValueError(
                f"selected workflow is missing from train inputs: {workflow_id}"
            )
        normalized.append(
            normalize_bundle(
                bundle,
                raw_root=args.raw_root,
                views_root=args.output_dir / "pilot20_views",
            )
        )

    queue = build_review_queue(normalized)
    summary = build_summary(normalized, queue)
    inputs_path = args.output_dir / "pilot20_inputs.jsonl"
    queue_path = args.output_dir / "pilot20_review_queue.jsonl"
    summary_path = args.output_dir / "pilot20_parse_summary.json"
    report_path = args.output_dir / "pilot20_parse_report.md"
    write_jsonl(inputs_path, normalized)
    write_jsonl(queue_path, queue)
    atomic_write_text(
        summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    atomic_write_text(report_path, render_report(summary))

    generated_view_paths = sorted(
        {
            Path(view["file_path"])
            for bundle in normalized
            for evidence in bundle["evidence"]
            for view in evidence["views"]
            if "file_path" in view
        },
        key=lambda path: path.as_posix(),
    )
    manifest = {
        "stage": "pilot20_normalization",
        "parser_version": PARSER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            args.selection.as_posix(): sha256_file(args.selection),
            args.train_inputs.as_posix(): sha256_file(args.train_inputs),
        },
        "outputs": {
            path.as_posix(): sha256_file(path)
            for path in (
                inputs_path,
                queue_path,
                summary_path,
                report_path,
                *generated_view_paths,
            )
        },
        "summary": summary,
    }
    atomic_write_text(
        args.output_dir / "pilot20_stage_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    print(
        f"Normalized {summary['workflow_count']} workflows and "
        f"{summary['evidence_count']} evidence files; "
        f"review queue={summary['review_queue_count']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
