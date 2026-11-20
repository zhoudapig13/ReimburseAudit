"""Export the three fixed model results and all 23 per-case predictions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


def main():
    folder = Path("runs/three_model_comparison_20261020")
    qwen = json.loads((folder / "qwen/report.json").read_text())
    new = json.loads((folder / "new_model/report.json").read_text())
    labels = {
        r["workflow_id"]: r["target"]
        for r in map(
            json.loads,
            Path("data/labels_private/validation_reference_targets.jsonl")
            .read_text()
            .splitlines(),
        )
    }
    names = [
        ("微调前：Qwen3-VL-8B-Instruct", qwen, "base"),
        ("微调后：Qwen3-VL-8B-Instruct + LoRA", qwen, "adapter"),
        (f"新增：{new['model']['name'].split('/')[-1]}（未微调）", new, "new_model"),
    ]
    summary = [
        [
            "模型",
            "样本数",
            "格式合格数",
            "报销判断正确数",
            "金额正确数",
            "判断与金额同时正确数",
            "同时正确率",
            "问题描述逐字一致数",
        ]
    ]
    details = [
        [
            "模型",
            "案例编号",
            "来源组",
            "标准FLAG",
            "标准金额",
            "预测FLAG",
            "预测金额",
            "格式合格",
            "判断正确",
            "金额正确",
            "同时正确",
            "问题描述逐字一致",
            "标准问题描述",
            "预测问题描述",
            "原始输出",
            "输入token数",
        ]
    ]
    wide = [
        [
            "案例编号",
            "标准FLAG",
            "标准金额",
            "原始Qwen_FLAG",
            "原始Qwen_金额",
            "原始Qwen_同时正确",
            "微调后_FLAG",
            "微调后_金额",
            "微调后_同时正确",
            "新增模型_FLAG",
            "新增模型_金额",
            "新增模型_同时正确",
        ]
    ]
    predictions = {}
    for name, report, variant in names:
        assert (
            report["validation_evaluated"] == 23
            and report["validation_available"] == 23
        )
        assert (
            not report["source_truncated"]
            and not report["test_targets_loaded"]
            and not report["deferred_workflow_ids"]
        )
        rows = [r for r in report["outputs"] if r["variant"] == variant]
        assert len(rows) == 23 and {r["workflow_id"] for r in rows} == set(labels)
        metrics = report[variant]
        summary.append(
            [
                name,
                23,
                metrics["schema_valid"],
                metrics["flag_correct"],
                metrics["amount_correct"],
                metrics["joint_flag_amount_correct"],
                metrics["joint_flag_amount_correct"] / 23,
                metrics["explanation_exact_match"],
            ]
        )
        predictions[variant] = {}
        for row in rows:
            target = labels[row["workflow_id"]]
            score = row["score"]
            try:
                prediction = json.loads(row["raw_output"])
                if not isinstance(prediction, dict):
                    prediction = {}
            except (ValueError, TypeError):
                prediction = {}
            joint = bool(
                score["schema_valid"]
                and score["flag_correct"]
                and score["amount_correct"]
            )
            flag = prediction.get("FLAG", "无法解析")
            amount = prediction.get("max_legal", "无法解析")
            predictions[variant][row["workflow_id"]] = [flag, str(amount), joint]
            details.append(
                [
                    name,
                    row["workflow_id"],
                    row["source_family_id"],
                    target["reference_flag"],
                    target["reference_max_legal"],
                    flag,
                    str(amount),
                    score["schema_valid"],
                    score["flag_correct"],
                    score["amount_correct"],
                    joint,
                    score["explanation_exact_match"],
                    json.dumps(target["problem_reports"], ensure_ascii=False),
                    json.dumps(
                        prediction.get("problem_reports", []), ensure_ascii=False
                    ),
                    row["raw_output"],
                    row["input_tokens"],
                ]
            )
        assert (
            sum(row[2] for row in predictions[variant].values())
            == metrics["joint_flag_amount_correct"]
        )
    for workflow_id, target in labels.items():
        wide.append(
            [
                workflow_id,
                target["reference_flag"],
                target["reference_max_legal"],
                *predictions["base"][workflow_id],
                *predictions["adapter"][workflow_id],
                *predictions["new_model"][workflow_id],
            ]
        )
    notes = [
        ["项目", "说明"],
        [
            "评测集合",
            "同一批23例验证案例，均未参与梯度更新；验证集已用于选择微调权重，分数不等同于独立测试成绩。",
        ],
        ["微调权重", qwen["adapter_checkpoint"]],
        [
            "选择规则",
            "本次对比开始前，固定使用此前验证集选出的最好微调权重；最新轮次训练结果另存原训练目录。",
        ],
        ["新模型", new["model"]["name"] + "，官方权重，不进行本项目微调。"],
        ["新模型固定版本", new["model"]["revision"]],
        [
            "输入",
            "同一份完整申请与材料；相同审计指令、表格与JSON无损紧凑序列化、图片最高1048576像素。模型各用其原生处理器和聊天模板。",
        ],
        [
            "生成",
            "均为贪心解码。Qwen最多生成1024 token；GLM按官方示例给8192 token总预算，最终答案上限1024 token。GLM只提取原生answer通道，不强制结束思考，不修补答案，不做格式重试或提示词调优。",
        ],
        [
            "GLM完整原始生成",
            "GLM的思考和最终答案完整保存在new_model/predictions.jsonl的raw_generation字段；表内原始输出是用于评分的最终答案文本。",
        ],
        ["FLAG", "1表示可全额报销，0表示不能全额报销；0可能对应部分报销或零报销。"],
        ["金额", "按精确数值相等评分，100与100.00相同。"],
        ["同时正确", "格式合格、FLAG正确、金额正确三个条件同时成立。"],
        ["问题描述", "仅做文本列表逐字匹配，未计算官方按问题点加权的语义评分。"],
        ["失败处理", "无法解析、字段错误或格式错误均保留原始输出，按统一规则计分。"],
        ["训练停止", "10月20日接近15点的一轮于14:57完成评测，随后训练已停止。"],
    ]
    for filename, rows in [
        ("三模型指标汇总.csv", summary),
        ("三模型逐例对照.csv", wide),
        ("三模型完整预测.csv", details),
    ]:
        with (folder / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            csv.writer(stream).writerows(rows)
    wb = Workbook()
    wb.remove(wb.active)
    for title, rows in [
        ("指标汇总", summary),
        ("23例并排对照", wide),
        ("完整预测与答案", details),
        ("评测说明", notes),
    ]:
        ws = wb.create_sheet(title)
        for row in rows:
            ws.append(row)
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="263B65")
        for i in range(1, ws.max_column + 1):
            ws.column_dimensions[get_column_letter(i)].width = 22
        ws.column_dimensions["A"].width = 44 if title != "23例并排对照" else 18
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        if title == "指标汇总":
            for row in ws.iter_rows(min_row=2):
                row[6].number_format = "0.0%"
        if title == "完整预测与答案":
            for col in ["M", "N", "O"]:
                ws.column_dimensions[col].width = 65
        if title == "评测说明":
            ws.column_dimensions["B"].width = 105
    wb.save(folder / "三个模型评测对比.xlsx")
    result = {
        "models": [
            {"name": name, "metrics": report[variant]}
            for name, report, variant in names
        ],
        "validation_workflows": 23,
        "xlsx": str((folder / "三个模型评测对比.xlsx").resolve()),
    }
    (folder / "comparison_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
