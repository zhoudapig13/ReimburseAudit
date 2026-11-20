"""Build leakage-aware source families and dataset splits for ReimburseBench."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
SPLIT_NAMES = tuple(SPLIT_RATIOS)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
POLICY_KEYWORDS = {
    "exchange_rate",
    "exchange rate",
    "guideline",
    "handbook",
    "policy",
    "procedure",
    "regulation",
    "reimbursement",
    "rule",
    "standard",
}


def workflow_sort_key(workflow_id: str) -> tuple[Any, ...]:
    """Sort dotted numeric workflow IDs naturally, with a string fallback."""
    parts: list[Any] = []
    for part in workflow_id.split("."):
        parts.append(int(part) if part.isdigit() else part)
    return tuple(parts)


def scenario_prefix(workflow_id: str) -> str:
    """Return the first two dotted components used by the release variants."""
    parts = workflow_id.split(".")
    if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]):
        raise ValueError(f"invalid workflow ID: {workflow_id!r}")
    return ".".join(parts[:2])


class UnionFind:
    """Minimal deterministic union-find for workflow linkage."""

    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if workflow_sort_key(left_root) <= workflow_sort_key(right_root):
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def valid_file_records(row: dict[str, Any], category: str) -> list[dict[str, Any]]:
    """Return existing, hashed file records in one manifest category."""
    return [
        file_record
        for file_record in row.get("files", [])
        if file_record.get("category") == category
        and file_record.get("exists", True)
        and file_record.get("sha256")
    ]


def build_source_families(
    rows: list[dict[str, Any]],
    *,
    visual_links: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Link same-prefix variants and verified cross-workflow receipt reuse."""
    workflow_ids = [str(row["workflow_id"]) for row in rows]
    if len(workflow_ids) != len(set(workflow_ids)):
        raise ValueError("manifest contains duplicate workflow IDs")

    union_find = UnionFind(workflow_ids)
    by_prefix: dict[str, list[str]] = defaultdict(list)
    receipt_hashes: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        workflow_id = str(row["workflow_id"])
        by_prefix[scenario_prefix(workflow_id)].append(workflow_id)
        for file_record in valid_file_records(row, "receipt"):
            receipt_hashes[str(file_record["sha256"])].append(workflow_id)

    for members in by_prefix.values():
        for workflow_id in members[1:]:
            union_find.union(members[0], workflow_id)

    for members in receipt_hashes.values():
        unique_members = sorted(set(members), key=workflow_sort_key)
        for workflow_id in unique_members[1:]:
            union_find.union(unique_members[0], workflow_id)

    visual_links = visual_links or []
    for link in visual_links:
        left_workflow = str(link["left_workflow"])
        right_workflow = str(link["right_workflow"])
        if (
            left_workflow not in union_find.parent
            or right_workflow not in union_find.parent
        ):
            raise ValueError(
                "visual link refers to a workflow absent from the manifest"
            )
        union_find.union(left_workflow, right_workflow)

    components: dict[str, list[str]] = defaultdict(list)
    for workflow_id in workflow_ids:
        components[union_find.find(workflow_id)].append(workflow_id)

    sorted_components = sorted(
        (sorted(members, key=workflow_sort_key) for members in components.values()),
        key=lambda members: workflow_sort_key(members[0]),
    )

    families: list[dict[str, Any]] = []
    workflow_to_family: dict[str, str] = {}
    for index, members in enumerate(sorted_components, start=1):
        family_id = f"sf_{index:03d}"
        member_set = set(members)
        prefixes = sorted(
            {scenario_prefix(workflow_id) for workflow_id in members},
            key=workflow_sort_key,
        )
        receipt_links = []
        for sha256, linked_workflows in sorted(receipt_hashes.items()):
            linked = sorted(set(linked_workflows) & member_set, key=workflow_sort_key)
            if len(linked) > 1:
                receipt_links.append({"sha256": sha256, "workflows": linked})
        family_visual_links = [
            link
            for link in visual_links
            if link["left_workflow"] in member_set
            and link["right_workflow"] in member_set
        ]
        family = {
            "source_family_id": family_id,
            "workflow_count": len(members),
            "workflows": members,
            "scenario_prefixes": prefixes,
            "linking_evidence": {
                "same_scenario_prefix": prefixes,
                "exact_receipt_hashes": receipt_links,
                "strong_visual_receipt_matches": family_visual_links,
            },
        }
        families.append(family)
        for workflow_id in members:
            workflow_to_family[workflow_id] = family_id

    return families, workflow_to_family


def family_features(
    family: dict[str, Any], rows_by_id: dict[str, dict[str, Any]]
) -> Counter[str]:
    """Build additive balancing features for one source family."""
    features: Counter[str] = Counter()
    for workflow_id in family["workflows"]:
        row = rows_by_id[workflow_id]
        features["__size__"] += 1
        features[f"difficulty:{row.get('difficulty', 'unknown')}"] += 1
        features[f"flag:{row.get('reference_flag', 'unknown')}"] += 1
        for logic_type in sorted(set(row.get("expected_types", []))):
            features[f"logic:{logic_type}"] += 1
    return features


def split_objective(
    assignments: dict[str, str],
    feature_by_family: dict[str, Counter[str]],
    global_features: Counter[str],
) -> float:
    """Score size and label-distribution deviation from requested ratios."""
    counts = {split_name: Counter() for split_name in SPLIT_NAMES}
    for family_id, split_name in assignments.items():
        counts[split_name].update(feature_by_family[family_id])

    score = 0.0
    for split_name, ratio in SPLIT_RATIOS.items():
        for feature, total in global_features.items():
            target = total * ratio
            difference = counts[split_name][feature] - target
            if feature == "__size__":
                weight = 24.0
            elif feature.startswith("difficulty:"):
                weight = 4.0
            elif feature.startswith("flag:"):
                weight = 2.0
            else:
                weight = 0.35
            score += weight * (difference / max(1.0, target)) ** 2

            if (
                split_name != "train"
                and feature.startswith("logic:")
                and total >= 5
                and counts[split_name][feature] == 0
            ):
                score += 2.0
    return score


def assign_splits(
    families: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, str]:
    """Assign whole source families with deterministic greedy + local search."""
    if len(families) < len(SPLIT_NAMES):
        raise ValueError("at least three source families are required")

    rows_by_id = {str(row["workflow_id"]): row for row in rows}
    feature_by_family = {
        family["source_family_id"]: family_features(family, rows_by_id)
        for family in families
    }
    global_features: Counter[str] = Counter()
    for features in feature_by_family.values():
        global_features.update(features)

    rng = random.Random(seed)
    ordered_families = list(families)
    rng.shuffle(ordered_families)
    ordered_families.sort(key=lambda item: -int(item["workflow_count"]))

    assignments: dict[str, str] = {}
    split_tie_order = list(SPLIT_NAMES)
    rng.shuffle(split_tie_order)
    for family in ordered_families:
        family_id = family["source_family_id"]
        best_split = min(
            split_tie_order,
            key=lambda split_name: (
                split_objective(
                    assignments | {family_id: split_name},
                    feature_by_family,
                    global_features,
                ),
                split_tie_order.index(split_name),
            ),
        )
        assignments[family_id] = best_split

    for split_name in SPLIT_NAMES:
        if split_name in assignments.values():
            continue
        donor = max(
            (
                family_id
                for family_id, current_split in assignments.items()
                if list(assignments.values()).count(current_split) > 1
            ),
            key=lambda family_id: feature_by_family[family_id]["__size__"],
        )
        assignments[donor] = split_name

    family_ids = sorted(assignments)
    current_score = split_objective(assignments, feature_by_family, global_features)
    for _ in range(50):
        improved = False
        for family_id in family_ids:
            original_split = assignments[family_id]
            for candidate_split in SPLIT_NAMES:
                if candidate_split == original_split:
                    continue
                if list(assignments.values()).count(original_split) <= 1:
                    continue
                proposal = assignments | {family_id: candidate_split}
                proposal_score = split_objective(
                    proposal, feature_by_family, global_features
                )
                if proposal_score + 1e-12 < current_score:
                    assignments = proposal
                    current_score = proposal_score
                    improved = True
                    original_split = candidate_split
        for left_index, left_id in enumerate(family_ids):
            for right_id in family_ids[left_index + 1 :]:
                if assignments[left_id] == assignments[right_id]:
                    continue
                proposal = dict(assignments)
                proposal[left_id], proposal[right_id] = (
                    proposal[right_id],
                    proposal[left_id],
                )
                proposal_score = split_objective(
                    proposal, feature_by_family, global_features
                )
                if proposal_score + 1e-12 < current_score:
                    assignments = proposal
                    current_score = proposal_score
                    improved = True
        if not improved:
            break

    workflow_splits: dict[str, str] = {}
    for family in families:
        split_name = assignments[family["source_family_id"]]
        for workflow_id in family["workflows"]:
            workflow_splits[workflow_id] = split_name
    return workflow_splits


def validate_assignments(
    rows: list[dict[str, Any]],
    families: list[dict[str, Any]],
    workflow_splits: dict[str, str],
) -> list[str]:
    """Validate completeness, family atomicity, and exact-receipt isolation."""
    issues: list[str] = []
    workflow_ids = {str(row["workflow_id"]) for row in rows}
    if set(workflow_splits) != workflow_ids:
        missing = sorted(workflow_ids - set(workflow_splits), key=workflow_sort_key)
        extra = sorted(set(workflow_splits) - workflow_ids, key=workflow_sort_key)
        issues.append(f"assignment coverage mismatch; missing={missing}, extra={extra}")

    invalid_splits = sorted(set(workflow_splits.values()) - set(SPLIT_NAMES))
    if invalid_splits:
        issues.append(f"invalid split names: {invalid_splits}")

    for family in families:
        splits = {
            workflow_splits.get(workflow_id) for workflow_id in family["workflows"]
        }
        if len(splits) != 1:
            issues.append(
                f"source family {family['source_family_id']} crosses splits: "
                f"{sorted(str(value) for value in splits)}"
            )

    receipt_hash_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        workflow_id = str(row["workflow_id"])
        for file_record in valid_file_records(row, "receipt"):
            receipt_hash_splits[str(file_record["sha256"])].add(
                workflow_splits.get(workflow_id, "missing")
            )
    for sha256, splits in sorted(receipt_hash_splits.items()):
        if len(splits) > 1:
            issues.append(
                f"exact receipt hash {sha256} crosses splits: {sorted(splits)}"
            )
    return issues


def normalized_filename(path: str) -> str:
    """Normalize a file stem for weak, review-only filename matching."""
    stem = Path(path).stem.lower()
    return re.sub(r"[^a-z0-9]+", "", stem)


def is_probable_policy(locations: list[dict[str, Any]]) -> bool:
    """Recognize generic policy-like names; never use this to merge samples."""
    names = " ".join(str(item["relative_path"]).lower() for item in locations)
    return any(keyword in names for keyword in POLICY_KEYWORDS)


def image_dhash(path: Path) -> tuple[int, int, int] | None:
    """Return a 64-bit difference hash and image dimensions, or None on failure."""
    try:
        from PIL import Image, ImageOps

        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size
            grayscale = image.convert("L").resize((9, 8))
            pixels = list(grayscale.tobytes())
    except (OSError, ValueError):
        return None

    value = 0
    for row_index in range(8):
        offset = row_index * 9
        for column_index in range(8):
            value <<= 1
            value |= pixels[offset + column_index] > pixels[offset + column_index + 1]
    return value, width, height


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def strong_visual_receipt_links(
    rows: list[dict[str, Any]],
    repo_root: Path,
    *,
    mae_threshold: float = 0.012,
    gray_correlation_threshold: float = 0.94,
    edge_correlation_threshold: float = 0.90,
) -> list[dict[str, Any]]:
    """Find high-confidence receipt-template matches across scenario prefixes."""
    try:
        import numpy as np
        from PIL import Image, ImageOps
    except ImportError as error:
        raise RuntimeError(
            "strong visual matching requires numpy and Pillow"
        ) from error

    records: list[dict[str, Any]] = []
    for row in rows:
        workflow_id = str(row["workflow_id"])
        folder = str(row.get("folder", workflow_id))
        prefix = scenario_prefix(workflow_id)
        for file_record in valid_file_records(row, "receipt"):
            if str(file_record.get("extension", "")).lower() not in IMAGE_EXTENSIONS:
                continue
            path = repo_root / "workflow" / folder / str(file_record["relative_path"])
            try:
                with Image.open(path) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    width, height = image.size
                    image = image.resize((96, 96))
                    rgb = np.asarray(image, dtype=np.float32) / 255.0
            except (OSError, ValueError):
                continue

            gray = rgb.mean(axis=2)
            edge = np.hypot(
                np.diff(gray, axis=0, append=gray[-1:]),
                np.diff(gray, axis=1, append=gray[:, -1:]),
            )
            gray_centered = gray.ravel() - float(gray.mean())
            edge_centered = edge.ravel() - float(edge.mean())
            gray_norm = float(np.linalg.norm(gray_centered))
            edge_norm = float(np.linalg.norm(edge_centered))
            if gray_norm == 0.0 or edge_norm == 0.0:
                continue
            records.append(
                {
                    "workflow_id": workflow_id,
                    "scenario_prefix": prefix,
                    "relative_path": str(file_record["relative_path"]),
                    "sha256": str(file_record["sha256"]),
                    "aspect_ratio": width / max(1, height),
                    "rgb": rgb,
                    "gray_unit": gray_centered / gray_norm,
                    "edge_unit": edge_centered / edge_norm,
                }
            )

    links: list[dict[str, Any]] = []
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            if left["scenario_prefix"] == right["scenario_prefix"]:
                continue
            if left["sha256"] == right["sha256"]:
                continue
            ratio_delta = abs(left["aspect_ratio"] - right["aspect_ratio"]) / max(
                left["aspect_ratio"], right["aspect_ratio"]
            )
            if ratio_delta > 0.08:
                continue
            pixel_mae = float(np.mean(np.abs(left["rgb"] - right["rgb"])))
            if pixel_mae > mae_threshold:
                continue
            gray_correlation = float(np.dot(left["gray_unit"], right["gray_unit"]))
            edge_correlation = float(np.dot(left["edge_unit"], right["edge_unit"]))
            if (
                gray_correlation < gray_correlation_threshold
                or edge_correlation < edge_correlation_threshold
            ):
                continue
            links.append(
                {
                    "left_workflow": left["workflow_id"],
                    "right_workflow": right["workflow_id"],
                    "left_path": left["relative_path"],
                    "right_path": right["relative_path"],
                    "pixel_mae": round(pixel_mae, 6),
                    "gray_correlation": round(gray_correlation, 6),
                    "edge_correlation": round(edge_correlation, 6),
                }
            )
    return links


def build_review_candidates(
    rows: list[dict[str, Any]],
    workflow_to_family: dict[str, str],
    workflow_splits: dict[str, str],
    *,
    repo_root: Path | None,
    dhash_threshold: int = 4,
) -> list[dict[str, Any]]:
    """Build non-binding candidates from background reuse and similar receipts."""
    candidates: list[dict[str, Any]] = []
    background_hashes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    receipt_names: dict[str, list[dict[str, Any]]] = defaultdict(list)
    image_records: list[dict[str, Any]] = []

    for row in rows:
        workflow_id = str(row["workflow_id"])
        family_id = workflow_to_family[workflow_id]
        folder = str(row.get("folder", workflow_id))
        for file_record in valid_file_records(row, "background"):
            background_hashes[str(file_record["sha256"])].append(
                {
                    "workflow_id": workflow_id,
                    "source_family_id": family_id,
                    "relative_path": file_record["relative_path"],
                }
            )
        for file_record in valid_file_records(row, "receipt"):
            location = {
                "workflow_id": workflow_id,
                "source_family_id": family_id,
                "relative_path": file_record["relative_path"],
                "sha256": file_record["sha256"],
            }
            normalized = normalized_filename(str(file_record["relative_path"]))
            if normalized:
                receipt_names[normalized].append(location)
            if (
                repo_root is not None
                and str(file_record.get("extension", "")).lower() in IMAGE_EXTENSIONS
            ):
                path = (
                    repo_root / "workflow" / folder / str(file_record["relative_path"])
                )
                signature = image_dhash(path)
                if signature is not None:
                    dhash, width, height = signature
                    image_records.append(
                        location
                        | {
                            "dhash": dhash,
                            "width": width,
                            "height": height,
                        }
                    )

    for sha256, locations in sorted(background_hashes.items()):
        families = sorted({item["source_family_id"] for item in locations})
        if len(families) <= 1:
            continue
        classification = (
            "shared_policy_template"
            if is_probable_policy(locations)
            else "possible_shared_case_document"
        )
        splits = sorted({workflow_splits[item["workflow_id"]] for item in locations})
        candidates.append(
            {
                "signal": "exact_background_hash",
                "classification": classification,
                "sha256": sha256,
                "source_families": families,
                "workflows": sorted(
                    {item["workflow_id"] for item in locations},
                    key=workflow_sort_key,
                ),
                "locations": locations,
                "assigned_splits": splits,
                "cross_split": len(splits) > 1,
                "blocks_release": classification == "possible_shared_case_document",
                "automatic_action": "kept_separate",
                "review_note": (
                    "Generic policies may be reused without implying the same event."
                    if classification == "shared_policy_template"
                    else "Review whether this is event-specific evidence from one source case."
                ),
            }
        )

    for normalized, locations in sorted(receipt_names.items()):
        families = sorted({item["source_family_id"] for item in locations})
        hashes = {item["sha256"] for item in locations}
        if len(families) <= 1 or len(hashes) <= 1:
            continue
        splits = sorted({workflow_splits[item["workflow_id"]] for item in locations})
        candidates.append(
            {
                "signal": "normalized_receipt_filename",
                "classification": "possible_edited_receipt",
                "normalized_filename": normalized,
                "source_families": families,
                "workflows": sorted(
                    {item["workflow_id"] for item in locations},
                    key=workflow_sort_key,
                ),
                "locations": locations,
                "assigned_splits": splits,
                "cross_split": len(splits) > 1,
                "blocks_release": False,
                "automatic_action": "kept_separate",
                "review_note": "Same filename but different bytes; inspect before merging.",
            }
        )

    seen_image_pairs: set[tuple[str, str, str, str]] = set()
    for left_index, left in enumerate(image_records):
        for right in image_records[left_index + 1 :]:
            if left["source_family_id"] == right["source_family_id"]:
                continue
            if left["sha256"] == right["sha256"]:
                continue
            left_ratio = left["width"] / max(1, left["height"])
            right_ratio = right["width"] / max(1, right["height"])
            ratio_delta = abs(left_ratio - right_ratio) / max(left_ratio, right_ratio)
            if ratio_delta > 0.08:
                continue
            distance = hamming_distance(left["dhash"], right["dhash"])
            if distance > dhash_threshold:
                continue
            pair_key = (
                left["workflow_id"],
                str(left["relative_path"]),
                right["workflow_id"],
                str(right["relative_path"]),
            )
            if pair_key in seen_image_pairs:
                continue
            seen_image_pairs.add(pair_key)
            splits = sorted(
                {
                    workflow_splits[left["workflow_id"]],
                    workflow_splits[right["workflow_id"]],
                }
            )
            candidates.append(
                {
                    "signal": "receipt_perceptual_hash",
                    "classification": "possible_edited_receipt",
                    "dhash_distance": distance,
                    "dhash_threshold": dhash_threshold,
                    "source_families": sorted(
                        {left["source_family_id"], right["source_family_id"]}
                    ),
                    "workflows": [left["workflow_id"], right["workflow_id"]],
                    "locations": [
                        {key: value for key, value in left.items() if key != "dhash"},
                        {key: value for key, value in right.items() if key != "dhash"},
                    ],
                    "assigned_splits": splits,
                    "cross_split": len(splits) > 1,
                    "blocks_release": False,
                    "automatic_action": "kept_separate",
                    "review_note": "Low-resolution visual hash selects image-similarity candidates for source review.",
                }
            )

    candidates.sort(
        key=lambda item: (
            item["signal"],
            item.get("classification", ""),
            item["workflows"],
        )
    )
    for index, candidate in enumerate(candidates, start=1):
        candidate["candidate_id"] = f"review_{index:03d}"
    return candidates


def summarize_split(
    rows: list[dict[str, Any]], workflow_splits: dict[str, str]
) -> dict[str, Any]:
    """Summarize workflow, difficulty, flag, and logic distributions."""
    summary: dict[str, Any] = {}
    for split_name in SPLIT_NAMES:
        split_rows = [
            row
            for row in rows
            if workflow_splits[str(row["workflow_id"])] == split_name
        ]
        difficulty = Counter(
            str(row.get("difficulty", "unknown")) for row in split_rows
        )
        flags = Counter(str(row.get("reference_flag", "unknown")) for row in split_rows)
        logic = Counter(
            logic_type
            for row in split_rows
            for logic_type in set(row.get("expected_types", []))
        )
        summary[split_name] = {
            "workflow_count": len(split_rows),
            "actual_ratio": round(len(split_rows) / max(1, len(rows)), 6),
            "difficulty": dict(sorted(difficulty.items())),
            "reference_flag": dict(sorted(flags.items())),
            "logic_point_activation": dict(sorted(logic.items())),
        }
    return summary


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=False))
            stream.write("\n")


def render_report(summary: dict[str, Any]) -> str:
    """Render a compact Chinese audit report for the generated split."""
    counts = summary["counts"]
    lines = [
        "# ReimburseBench 来源分组与数据划分报告",
        "",
        "## 结论",
        "",
        f"- 保留全部 {counts['workflows']} 条 workflow，没有合并或删除训练样本。",
        f"- 实际观察到 {counts['observed_scenario_prefixes']} 个编号前缀；论文中的 44 指完整业务场景分类表，不强制等于仓库前缀数。",
        f"- 共建立 {counts['source_families']} 个来源组；同一来源组只进入一个分区。",
        f"- 发现并绑定 {counts['strong_visual_receipt_link_pairs']} 对高置信度票据模板变体。",
        f"- 划分状态：`{summary['status']}`。",
        "",
        "## 自动绑定规则",
        "",
        "1. 编号前两段相同的 workflow 视为同一场景下的变体，绑定到同一分区。",
        "2. 字节级完全相同的票据即使跨编号前缀，也绑定到同一来源组。",
        "3. 像素差、灰度相关性和边缘相关性同时达到严格阈值的票据模板变体会绑定。",
        "4. 相同政策或背景文档不自动绑定；仅输出为复核线索。",
        "5. 文件名相同但内容不同、或只有低分辨率感知哈希相近的票据不自动绑定。",
        "",
        "## 划分结果",
        "",
        "| 分区 | workflow 数 | 实际占比 | easy | medium | hard | flag=1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name in SPLIT_NAMES:
        item = summary["splits"][split_name]
        lines.append(
            "| {name} | {count} | {ratio:.1%} | {easy} | {medium} | {hard} | {flag} |".format(
                name=split_name,
                count=item["workflow_count"],
                ratio=item["actual_ratio"],
                easy=item["difficulty"].get("easy", 0),
                medium=item["difficulty"].get("medium", 0),
                hard=item["difficulty"].get("hard", 0),
                flag=item["reference_flag"].get("1", 0),
            )
        )

    review = summary["manual_review"]
    lines.extend(
        [
            "",
            "## 复核状态",
            "",
            f"- 候选总数：{review['candidate_count']}",
            f"- 当前跨分区候选：{review['cross_split_candidate_count']}",
            f"- 阻断发布的跨分区候选：{review['blocking_cross_split_candidate_count']}",
            "- 感知哈希筛选相似图片候选，同源关系结合原始票据与事件信息审核确认。",
            "- 只有事件专属背景材料跨分区时才阻断；通用政策、通用汇率表和弱图像信号保留为审计记录。",
            "",
            "## 完整性检查",
            "",
            f"- 检查结果：`{summary['validation']['status']}`",
            f"- 问题数：{len(summary['validation']['issues'])}",
            "- 已检查：每条 workflow 恰好一次、来源组不跨分区、完全相同票据不跨分区。",
            "",
            "## 文件说明",
            "",
            "- `source_groups.jsonl`：每个来源组的成员和绑定证据。",
            "- `split_assignments.jsonl`：每条 workflow 的最终暂定分区。",
            "- `group_review_candidates.jsonl`：需要人工查看的弱证据候选。",
            "- `split_summary.json`：规则、统计、检查结果的机器可读汇总。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--dhash-threshold", type=int, default=4)
    parser.add_argument("--visual-mae-threshold", type=float, default=0.012)
    parser.add_argument("--visual-gray-correlation", type=float, default=0.94)
    parser.add_argument("--visual-edge-correlation", type=float, default=0.90)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.manifest)
    rows.sort(key=lambda row: workflow_sort_key(str(row["workflow_id"])))
    visual_links = (
        strong_visual_receipt_links(
            rows,
            args.repo_root,
            mae_threshold=args.visual_mae_threshold,
            gray_correlation_threshold=args.visual_gray_correlation,
            edge_correlation_threshold=args.visual_edge_correlation,
        )
        if args.repo_root is not None
        else []
    )
    families, workflow_to_family = build_source_families(
        rows, visual_links=visual_links
    )
    workflow_splits = assign_splits(families, rows, seed=args.seed)
    issues = validate_assignments(rows, families, workflow_splits)

    for family in families:
        family["assigned_split"] = workflow_splits[family["workflows"][0]]

    candidates = build_review_candidates(
        rows,
        workflow_to_family,
        workflow_splits,
        repo_root=args.repo_root,
        dhash_threshold=args.dhash_threshold,
    )
    assignment_records = [
        {
            "workflow_id": str(row["workflow_id"]),
            "source_family_id": workflow_to_family[str(row["workflow_id"])],
            "split": workflow_splits[str(row["workflow_id"])],
            "scenario_prefix": scenario_prefix(str(row["workflow_id"])),
            "difficulty": row.get("difficulty"),
            "reference_flag": row.get("reference_flag"),
            "expected_types": row.get("expected_types", []),
        }
        for row in rows
    ]

    cross_split_candidates = [
        candidate for candidate in candidates if candidate["cross_split"]
    ]
    blocking_cross_split_candidates = [
        candidate for candidate in cross_split_candidates if candidate["blocks_release"]
    ]
    status = (
        "invalid"
        if issues
        else (
            "provisional_manual_review_required"
            if blocking_cross_split_candidates
            else ("ready_with_review_notes" if candidates else "ready")
        )
    )
    summary = {
        "schema_version": "1.0",
        "status": status,
        "seed": args.seed,
        "target_ratios": SPLIT_RATIOS,
        "visual_link_thresholds": {
            "pixel_mae_max": args.visual_mae_threshold,
            "gray_correlation_min": args.visual_gray_correlation,
            "edge_correlation_min": args.visual_edge_correlation,
            "aspect_ratio_relative_delta_max": 0.08,
        },
        "inputs": {
            "manifest": str(args.manifest),
            "repo_root": str(args.repo_root) if args.repo_root else None,
        },
        "grouping_rules": {
            "hard_links": [
                "same first-two-component workflow prefix",
                "exact SHA-256 reuse of a receipt across workflows",
                "strong receipt-template match meeting all visual thresholds",
            ],
            "not_hard_links": [
                "shared policy or background document alone",
                "normalized filename similarity alone",
                "receipt perceptual-hash similarity alone",
            ],
        },
        "counts": {
            "workflows": len(rows),
            "observed_scenario_prefixes": len(
                {scenario_prefix(str(row["workflow_id"])) for row in rows}
            ),
            "paper_taxonomy_scenarios": 44,
            "source_families": len(families),
            "strong_visual_receipt_link_pairs": len(visual_links),
        },
        "splits": summarize_split(rows, workflow_splits),
        "manual_review": {
            "candidate_count": len(candidates),
            "cross_split_candidate_count": len(cross_split_candidates),
            "blocking_cross_split_candidate_count": len(
                blocking_cross_split_candidates
            ),
            "by_signal": dict(
                sorted(Counter(item["signal"] for item in candidates).items())
            ),
            "cross_split_by_signal": dict(
                sorted(
                    Counter(item["signal"] for item in cross_split_candidates).items()
                )
            ),
        },
        "validation": {
            "status": "passed" if not issues else "failed",
            "issues": issues,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "source_groups.jsonl", families)
    write_jsonl(args.output_dir / "split_assignments.jsonl", assignment_records)
    write_jsonl(args.output_dir / "group_review_candidates.jsonl", candidates)
    (args.output_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (args.output_dir / "split_report.md").write_text(
        render_report(summary),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
