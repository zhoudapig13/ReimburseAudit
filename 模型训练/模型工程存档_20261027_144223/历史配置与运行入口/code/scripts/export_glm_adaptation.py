"""Keep the original three-model results and add frozen GLM adaptation results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[2]


def export(folder):
    previous = ROOT / "runs/three_model_comparison_20261020"
    qwen = json.loads((previous / "qwen/report.json").read_text())
    original = json.loads((previous / "new_model/report.json").read_text())
    adapted = json.loads((folder / "report.json").read_text())
    labels = {
        r["workflow_id"]: r["target"]
        for r in map(
            json.loads,
            (ROOT / "data/labels_private/validation_reference_targets.jsonl")
            .read_text()
            .splitlines(),
        )
    }
    models = [
        ("原始Qwen3-VL-8B", qwen, "base", "base"),
        ("微调后Qwen3-VL-8B（最好权重）", qwen, "adapter", "adapter"),
        ("GLM原始推理（未微调）", original, "new_model", "new_model"),
        ("GLM格式适配后（同一权重）", adapted, "adapted_glm", "glm_adapted"),
    ]
    summary = [
        [
            "版本",
            "样本数",
            "格式合格",
            "报销判断正确",
            "金额正确",
            "两项同时正确",
            "同时正确率",
        ]
    ]
    details = [
        [
            "版本",
            "案例编号",
            "标准FLAG",
            "标准金额",
            "预测FLAG",
            "预测金额",
            "格式合格",
            "判断正确",
            "金额正确",
            "同时正确",
            "原始最终答案",
        ]
    ]
    comparison = [
        [
            "案例编号",
            "标准FLAG",
            "标准金额",
            *[name + "_同时正确" for name, _, _, _ in models],
        ]
    ]
    by_model = {}
    for name, report, key, variant in models:
        assert (
            report["validation_evaluated"] == 23
            and not report["source_truncated"]
            and not report["test_targets_loaded"]
        )
        rows = [r for r in report["outputs"] if r["variant"] == variant]
        assert len(rows) == 23 and {r["workflow_id"] for r in rows} == set(labels)
        metric = report[key]
        summary.append(
            [
                name,
                23,
                metric["schema_valid"],
                metric["flag_correct"],
                metric["amount_correct"],
                metric["joint_flag_amount_correct"],
                metric["joint_flag_amount_correct"] / 23,
            ]
        )
        by_model[name] = {}
        for row in rows:
            target = labels[row["workflow_id"]]
            score = row["score"]
            try:
                prediction = json.loads(row["raw_output"])
                if not isinstance(prediction, dict):
                    prediction = {}
            except ValueError:
                prediction = {}
            joint = bool(
                score["schema_valid"]
                and score["flag_correct"]
                and score["amount_correct"]
            )
            by_model[name][row["workflow_id"]] = joint
            details.append(
                [
                    name,
                    row["workflow_id"],
                    target["reference_flag"],
                    target["reference_max_legal"],
                    str(prediction.get("FLAG", "无法解析")),
                    str(prediction.get("max_legal", "无法解析")),
                    score["schema_valid"],
                    score["flag_correct"],
                    score["amount_correct"],
                    joint,
                    row["raw_output"],
                ]
            )
        assert sum(by_model[name].values()) == metric["joint_flag_amount_correct"]
    for wid, target in labels.items():
        comparison.append(
            [
                wid,
                target["reference_flag"],
                target["reference_max_legal"],
                *[by_model[name][wid] for name, _, _, _ in models],
            ]
        )
    notes = [
        ["项目", "说明"],
        ["数据", "同一批23例验证样本；验证集已用于选择Qwen微调权重。"],
        [
            "GLM适配",
            "模型权重不变；增加系统审计指令和材料末尾提醒，并使用JSON Schema约束最终答案的生成。",
        ],
        [
            "检查流程",
            "先用训练集3例检查格式，通过后冻结方案，再一次性评测23例；未用验证答案调整适配方式。",
        ],
        [
            "推理预算",
            "适配后思考最多4096 token，最终JSON最多1024 token；思考达到预算时关闭原生思考通道再生成JSON。",
        ],
        [
            "答案约束",
            "限定字段和类型、FLAG为0或1、金额为十进制字符串，问题列表最多6条且每条192字符。具体判断、金额和描述由模型生成。",
        ],
        [
            "评分",
            "沿用相同严格JSON格式、FLAG精确匹配和金额数值匹配；两项同时正确要求格式合格。",
        ],
        ["对照", "原始GLM和适配后的GLM分别列出；适配后增加了提示词和解码约束。"],
        [
            "原始输出",
            "predictions.jsonl保存最终答案、原始思考、注入的通道控制符和生成信息；未用规则修补模型答案。",
        ],
        ["Qwen最好权重", qwen["adapter_checkpoint"]],
    ]
    wb = Workbook()
    wb.remove(wb.active)
    for title, rows in [
        ("指标汇总", summary),
        ("23例对照", comparison),
        ("92条预测明细", details),
        ("适配说明", notes),
    ]:
        with (folder / f"{title}.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            csv.writer(stream).writerows(rows)
        ws = wb.create_sheet(title)
        for row in rows:
            ws.append(row)
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="263B65")
        for i in range(1, ws.max_column + 1):
            ws.column_dimensions[get_column_letter(i)].width = 24
        ws.column_dimensions["A"].width = 40
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        if title == "指标汇总":
            for i, row in enumerate(ws.iter_rows(min_row=2), 2):
                row[6].number_format = "0.0%"
                ws.row_dimensions[i].height = 32
        if title == "92条预测明细":
            ws.column_dimensions["K"].width = 90
        if title == "适配说明":
            ws.column_dimensions["B"].width = 110
    wb.save(folder / "GLM适配前后与Qwen对比.xlsx")
    result = {
        "models": [
            {"name": name, "metrics": report[key]} for name, report, key, _ in models
        ],
        "validation_workflows": 23,
        "xlsx": str(folder / "GLM适配前后与Qwen对比.xlsx"),
    }
    (folder / "comparison_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
