"""Normalize every packed input in one dataset split without loading targets."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from normalize_pilot_evidence import (
    PARSER_VERSION,
    atomic_write_text,
    build_review_queue,
    build_summary,
    load_jsonl,
    normalize_bundle,
    sha256_file,
    write_jsonl,
)

DATASET_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def dataset_name(value: str) -> str:
    if not DATASET_NAME_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "invalid dataset name; use lowercase letters, digits, and underscores"
        )
    return value


def render_report(name: str, summary: dict[str, object]) -> str:
    return "\n".join(
        [
            f"# {name} 附件解析报告",
            "",
            "本阶段只读取输入附件，不加载或生成参考答案。图片、Word 内嵌图和扫描 PDF 页进入多模态读取队列。",
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
            "- `parsed`：本地解析器已取得结构化内容。",
            "- `parsed_with_warnings`：已取得结构化内容，关联图片读取或公式核验任务。",
            "- `visual_only`：保留原图及视觉读取入口。",
            "- `failed`：解析失败，必须进入失败队列。",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument(
        "--raw-root", type=Path, default=Path("data_raw/ReimburseBench")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/normalized_full"))
    parser.add_argument("--dataset-name", type=dataset_name, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundles = load_jsonl(args.inputs)
    views_root = args.output_dir / f"{args.dataset_name}_views"
    normalized = [
        normalize_bundle(bundle, raw_root=args.raw_root, views_root=views_root)
        for bundle in bundles
    ]
    queue = build_review_queue(normalized)
    summary = build_summary(normalized, queue)

    inputs_path = args.output_dir / f"{args.dataset_name}_inputs.jsonl"
    queue_path = args.output_dir / f"{args.dataset_name}_review_queue.jsonl"
    summary_path = args.output_dir / f"{args.dataset_name}_parse_summary.json"
    report_path = args.output_dir / f"{args.dataset_name}_parse_report.md"
    manifest_path = args.output_dir / f"{args.dataset_name}_stage_manifest.json"
    write_jsonl(inputs_path, normalized)
    write_jsonl(queue_path, queue)
    atomic_write_text(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(report_path, render_report(args.dataset_name, summary))

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
        "stage": "dataset_normalization",
        "dataset_name": args.dataset_name,
        "parser_version": PARSER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {args.inputs.as_posix(): sha256_file(args.inputs)},
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
        "targets_loaded": False,
    }
    atomic_write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(
        f"Normalized {summary['workflow_count']} {args.dataset_name} workflows and "
        f"{summary['evidence_count']} evidence files; "
        f"review queue={summary['review_queue_count']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
