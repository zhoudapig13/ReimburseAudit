# 报销审计专利实现工程

工程包含政策关系判断、证据贡献核验、局部校正三个神经部件，以及整单答案的LoRA训练和评测代码。本次交付的LoRA权重对应Qwen3-VL-8B-Instruct整单审核答案生成；专利技术方案见上一级“01_专利文档”。

**2026-10-27 三模块续训结果：12/23（52.17%）**，FLAG与最大合规金额同时正确。已使用材料读取LoRA继续微调，并刷新全部特征、重训三个核心模块。最新版本见 [当前最佳三模块索引](models/当前最佳三模块.json)。该成绩来自调参验证集，独立测试集未评测；下文整单答案LoRA属于历史运行。

三个模块的统一训练、推理和评测方法见[三模块快速运行](三模块快速运行.md)。入口为 `code/scripts/patent_pipeline.py`，短训练配置为 `configs/patent_fast.yaml`。

## 文件位置

| 内容 | 路径 |
|---|---|
| 神经部件与训练逻辑 | `audit_engine/` |
| 数据处理、训练和评测入口 | `code/scripts/` |
| 配置与阈值 | `configs/` |
| 程序测试 | `tests/` |
| 最终LoRA权重及续训状态 | `models/报销审计_LoRA/` |
| 原始公开数据及许可 | `data_raw/ReimburseBench/` |
| 标准化输入与固定划分 | `data/normalized_full/`、`data/splits/` |
| 分区参考答案 | `data/labels_private/` |
| 训练与评测记录 | `runs/` |
| 精确版本记录 | `environment.lock.json`、`models/模型版本.json` |

## 环境

所有命令在本目录执行。使用Python 3.12；原运行环境为Linux、NVIDIA PyTorch 2.8.0a0、RTX5090，精确版本见锁文件。在保留服务器CUDA/PyTorch环境的基础上创建环境并安装依赖：

```bash
python3.12 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.lock.txt
```

锁文件中的NVIDIA PyTorch构建来自原服务器镜像。`torchao`原版本记录在`environment.lock.json`中，随服务器镜像保留。

## 检查工程

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests
```

## 加载与评测

交付的模型文件为`models/报销审计_LoRA/`中的LoRA checkpoint及续训状态。加载时将`Qwen/Qwen3-VL-8B-Instruct`的固定版本`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`另行准备到`models/Qwen3-VL-8B-Instruct/`，再执行下列评测命令。

```bash
.venv/bin/python code/scripts/evaluate_official_adapter.py \
  --adapter models/报销审计_LoRA \
  --config configs/official_sft5090_corrected.yaml \
  --device cuda:3 --token-budget 131072 \
  --output-dir runs/reproduction_qwen
```

该入口在同一批23例验证材料上比较底座与LoRA；输入保留全部材料。参考答案在预测完成后读入。测试集保持原划分。

GLM对照的配置及输出记录保存在`runs/glm_format_adaptation_20261021_attempt2/`，模型版本记录在`models/模型版本.json`。

原件与数据许可保留在`data_raw/ReimburseBench/`，数据集来源版本为`71476774f25bfc022ae926461c3e76ed9841fc48`。原始数据及来源分组保留，派生解析结果可通过`pack_training_data.py`和`normalize_dataset_evidence.py`重建。参考答案目录仅供训练及离线评测使用。

历史报告中的绝对路径记录当时的运行位置；本次整理后的加载路径以本README和`models/模型版本.json`为准。
