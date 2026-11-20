"""Select a source-diverse training pilot without exposing labels to model inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

SELECTION_VERSION = "pilot_selection_v1"


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


def _stable_tiebreak(seed: int, workflow_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{workflow_id}".encode()).hexdigest()
    return int(digest, 16)


def _proportional_quotas(values: list[Any], limit: int) -> dict[Any, int]:
    counts = Counter(values)
    total = sum(counts.values())
    if not total:
        return {}
    exact = {key: limit * count / total for key, count in counts.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = limit - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda key: (exact[key] - quotas[key], counts[key], str(key)),
        reverse=True,
    )
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def select_pilot_workflows(
    assignments: list[dict[str, Any]],
    bundles: list[dict[str, Any]],
    *,
    limit: int,
    seed: int,
    required_workflow_ids: list[str] | tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Choose at most one training workflow per source family."""
    if limit <= 0:
        raise ValueError("pilot limit must be positive")
    bundles_by_id = {str(row["workflow_id"]): row for row in bundles}
    candidates = []
    for assignment in assignments:
        if assignment.get("split") != "train":
            continue
        workflow_id = str(assignment.get("workflow_id", ""))
        family_id = str(assignment.get("source_family_id", ""))
        if not workflow_id or not family_id:
            raise ValueError("train assignment is missing workflow or family ID")
        bundle = bundles_by_id.get(workflow_id)
        if bundle is None:
            raise ValueError(f"missing packed train input for {workflow_id}")
        evidence_types = sorted(
            {str(item.get("type", "")) for item in bundle.get("evidence", [])}
        )
        candidates.append(
            {
                "workflow_id": workflow_id,
                "source_family_id": family_id,
                "difficulty": str(assignment.get("difficulty", "unknown")),
                "reference_flag": assignment.get("reference_flag"),
                "expected_types": sorted(
                    {str(value) for value in assignment.get("expected_types", [])}
                ),
                "evidence_types": evidence_types,
            }
        )

    family_count = len({row["source_family_id"] for row in candidates})
    if family_count < limit:
        raise ValueError(
            f"only {family_count} train source families are available for "
            f"a {limit}-workflow pilot"
        )

    difficulty_quota = _proportional_quotas(
        [row["difficulty"] for row in candidates], limit
    )
    flag_quota = _proportional_quotas(
        [row["reference_flag"] for row in candidates], limit
    )
    logic_frequency = Counter(
        logic_type for row in candidates for logic_type in row["expected_types"]
    )

    selected = []
    selected_families: set[str] = set()
    selected_logic_types: set[str] = set()
    selected_evidence_types: set[str] = set()
    difficulty_counts: Counter[str] = Counter()
    flag_counts: Counter[Any] = Counter()

    candidates_by_id = {row["workflow_id"]: row for row in candidates}
    for workflow_id in required_workflow_ids:
        if len(selected) >= limit:
            raise ValueError("required workflows exceed the pilot limit")
        chosen = candidates_by_id.get(workflow_id)
        if chosen is None:
            raise ValueError(
                f"required workflow is not available in train: {workflow_id}"
            )
        if chosen["source_family_id"] in selected_families:
            raise ValueError(
                "required workflows cannot share a source family: "
                f"{chosen['source_family_id']}"
            )
        selected.append(chosen)
        selected_families.add(chosen["source_family_id"])
        selected_logic_types.update(chosen["expected_types"])
        selected_evidence_types.update(chosen["evidence_types"])
        difficulty_counts[chosen["difficulty"]] += 1
        flag_counts[chosen["reference_flag"]] += 1

    while len(selected) < limit:
        eligible = [
            row
            for row in candidates
            if row["source_family_id"] not in selected_families
        ]
        if not eligible:
            raise ValueError("pilot selection exhausted eligible source families")

        def score(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
            difficulty_need = max(
                difficulty_quota.get(row["difficulty"], 0)
                - difficulty_counts[row["difficulty"]],
                0,
            )
            flag_need = max(
                flag_quota.get(row["reference_flag"], 0)
                - flag_counts[row["reference_flag"]],
                0,
            )
            new_logic = set(row["expected_types"]) - selected_logic_types
            rare_logic_score = sum(
                max(len(candidates) - logic_frequency[value], 1) for value in new_logic
            )
            new_evidence_count = len(
                set(row["evidence_types"]) - selected_evidence_types
            )
            return (
                difficulty_need,
                flag_need,
                new_evidence_count,
                rare_logic_score,
                -_stable_tiebreak(seed, row["workflow_id"]),
            )

        chosen = max(eligible, key=score)
        selected.append(chosen)
        selected_families.add(chosen["source_family_id"])
        selected_logic_types.update(chosen["expected_types"])
        selected_evidence_types.update(chosen["evidence_types"])
        difficulty_counts[chosen["difficulty"]] += 1
        flag_counts[chosen["reference_flag"]] += 1

    return [
        {
            "selection_version": SELECTION_VERSION,
            "selection_rank": rank,
            "workflow_id": row["workflow_id"],
            "source_family_id": row["source_family_id"],
            "input_ref": "data/packed/train_inputs.jsonl",
            "evidence_types": row["evidence_types"],
        }
        for rank, row in enumerate(selected, start=1)
    ]


def build_summary(
    selection: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    *,
    seed: int,
    required_workflow_ids: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    assignment_by_id = {
        str(row["workflow_id"]): row
        for row in assignments
        if row.get("split") == "train"
    }
    selected_assignments = [assignment_by_id[row["workflow_id"]] for row in selection]
    return {
        "selection_version": SELECTION_VERSION,
        "seed": seed,
        "required_workflow_ids": list(required_workflow_ids),
        "workflow_count": len(selection),
        "source_family_count": len({row["source_family_id"] for row in selection}),
        "difficulty_counts": dict(
            sorted(
                Counter(
                    str(row.get("difficulty", "unknown"))
                    for row in selected_assignments
                ).items()
            )
        ),
        "reference_flag_counts": {
            str(key): value
            for key, value in sorted(
                Counter(
                    row.get("reference_flag") for row in selected_assignments
                ).items(),
                key=lambda item: str(item[0]),
            )
        },
        "logic_type_counts": dict(
            sorted(
                Counter(
                    value
                    for row in selected_assignments
                    for value in row.get("expected_types", [])
                ).items()
            )
        ),
        "evidence_type_counts": dict(
            sorted(
                Counter(
                    value for row in selection for value in row["evidence_types"]
                ).items()
            )
        ),
    }


def render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# 20 个训练工作流试点选择报告",
            "",
            "本文件记录抽样结果，运行输入由材料处理入口单独生成。选择规则为：仅从 Train 中抽取，每个来源组最多一个工作流，并尽量平衡难度、原始 Flag、逻辑类型和附件格式。",
            "",
            f"- 工作流：{summary['workflow_count']}",
            f"- 独立来源组：{summary['source_family_count']}",
            f"- 固定随机种子：{summary['seed']}",
            f"- 指南指定必选：{', '.join(summary['required_workflow_ids']) or '无'}",
            f"- 难度分布：{json.dumps(summary['difficulty_counts'], ensure_ascii=False)}",
            f"- 原始 Flag 分布：{json.dumps(summary['reference_flag_counts'], ensure_ascii=False)}",
            f"- 附件格式覆盖：{json.dumps(summary['evidence_type_counts'], ensure_ascii=False)}",
            "",
            "`pilot20_selection.jsonl` 保存工作流、来源组和附件格式，标签分布用于工程抽样统计。运行材料与监督目标分别保存。",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assignments", type=Path, default=Path("data/splits/split_assignments.jsonl")
    )
    parser.add_argument(
        "--train-inputs", type=Path, default=Path("data/packed/train_inputs.jsonl")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/pilot"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--include-workflow",
        action="append",
        default=["1.1.1"],
        help="Training workflow that must be included; may be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assignments = load_jsonl(args.assignments)
    bundles = load_jsonl(args.train_inputs)
    selection = select_pilot_workflows(
        assignments,
        bundles,
        limit=args.limit,
        seed=args.seed,
        required_workflow_ids=args.include_workflow,
    )
    summary = build_summary(
        selection,
        assignments,
        seed=args.seed,
        required_workflow_ids=args.include_workflow,
    )
    write_jsonl(args.output_dir / "pilot20_selection.jsonl", selection)
    atomic_write_text(
        args.output_dir / "pilot20_selection_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write_text(
        args.output_dir / "pilot20_selection_report.md", render_report(summary)
    )
    print(
        f"Selected {summary['workflow_count']} workflows from "
        f"{summary['source_family_count']} train source families."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
