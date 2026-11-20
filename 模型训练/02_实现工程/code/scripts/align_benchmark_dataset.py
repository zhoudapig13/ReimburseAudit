"""Align one normalized training dataset with benchmark reference targets."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from align_pilot_benchmark import (
    ALIGNMENT_VERSION,
    _index_by_workflow,
    build_alignment_record,
    build_benchmark_pair,
    build_summary,
    load_jsonl,
    sha256_file,
    validate_pair_coverage,
    write_jsonl,
)
from normalize_dataset_evidence import dataset_name
from normalize_pilot_evidence import atomic_write_text


def render_report(
    name: str,
    summary: dict[str, object],
    hold_records: list[dict[str, object]],
) -> str:
    hold_lines = (
        "\n".join(
            f"- `{row['workflow_id']}`：{', '.join(row['hold_reasons'])}"
            for row in hold_records
        )
        or "- 无"
    )
    return f"""# {name} Benchmark 对齐报告

## 结果

- 已对齐 {summary["workflow_count"]} 个训练工作流。
- 可生成端到端 Benchmark 监督配对：{summary["benchmark_pair_count"]} 个。
- 目标泄漏：{summary["target_leakage_count"]} 个。
- FLAG 算术不一致：{summary["flag_amount_mismatch_count"]} 个。
- 数字来源检查：{json.dumps(summary["numeric_traceability_counts"], ensure_ascii=False, sort_keys=True)}。

## 暂停样本

{hold_lines}

## 监督用途

- 只加载训练集参考答案。
- 配对可用于端到端 Benchmark 监督实验。
- 条件、关系、引用、三分金额和读取SFT标签分别通过原始材料及政策核验后形成。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/training_full"))
    parser.add_argument("--dataset-name", type=dataset_name, required=True)
    return parser.parse_args()


def main() -> int:
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
    prefix = f"{args.dataset_name}_benchmark"
    alignment_path = args.output_dir / f"{prefix}_alignment.jsonl"
    pairs_path = args.output_dir / f"{prefix}_pairs.jsonl"
    hold_path = args.output_dir / f"{prefix}_hold_queue.jsonl"
    summary_path = args.output_dir / f"{prefix}_summary.json"
    report_path = args.output_dir / f"{prefix}_report.md"
    manifest_path = args.output_dir / f"{prefix}_stage_manifest.json"
    write_jsonl(alignment_path, alignment_records)
    write_jsonl(pairs_path, benchmark_pairs)
    write_jsonl(hold_path, hold_records)
    atomic_write_text(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(
        report_path, render_report(args.dataset_name, summary, hold_records)
    )

    manifest = {
        "stage": "benchmark_dataset_alignment",
        "dataset_name": args.dataset_name,
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
        "rollback": f"Remove {args.output_dir.as_posix()}/{prefix}_* outputs.",
        "validation_and_test_targets_loaded": False,
        "component_training_labels_generated": False,
        "reading_sft_labels_generated": False,
    }
    atomic_write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(
        f"Aligned {len(alignment_records)} {args.dataset_name} workflows; "
        f"pairs={len(benchmark_pairs)}, held={len(hold_records)}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
