# 报销审计三模块实现工程

已完成政策关系判断、证据贡献核验、局部校正三个模块的训练、加载及联合推理验证，并完成配套材料读取LoRA微调。
微调后三模块系统准确率为 **52.17%（12/23）**，按FLAG与最大合规金额同时正确计分。

交付模型索引：[当前最佳三模块](models/当前最佳三模块.json)。完整命令见[三模块快速运行](三模块快速运行.md)。

| 内容 | 交付路径 |
|---|---|
| 联合checkpoint与各阶段checkpoint | `models/报销审计_三模块/checkpoint_20261027_5217/` |
| 配套材料读取LoRA | `models/报销审计_三模块/checkpoint_20261027_5217/reader_lora/` |
| 运行配置 | `configs/patent_delivery.yaml` |
| 读取LoRA训练配置 | `configs/patent_reader_delivery.yaml` |
| 输入、监督、固定划分与真实底座特征 | `data/三模块交付_20261027/` |
| 原始材料与许可 | `data_raw/ReimburseBench/` |
| 三模块源码 | `audit_engine/` |
| 训练、编码、推理和评分入口 | `code/scripts/` |
| 工程测试 | `tests/` |
| 逐例结果与准确率 | `../03_评测结果/三模块评测/` |
| 模型版本与文件校验值 | `models/模型版本.json` |

三个模块与共享投影保存在同一个 `audit_heads.pt` 中。配套读取LoRA负责材料事实读取，三模块负责政策关系、证据贡献与局部校正。推理输出事实表F、金额表B、轨迹T及复核任务。

底座为 `Qwen/Qwen3-VL-8B-Instruct`，固定revision为 `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`。底座单独准备；环境精确版本见 `environment.lock.json`，依赖见 `requirements.lock.txt`。

工程已通过242项测试，1项跳过。23例完整推理中，关系模块调用462次、贡献模块调用55次、局部纠错模块调用32次。完整性核验与评分文件随交付提供。
