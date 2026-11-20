# ReimburseAudit

## Policy-Aware Reimbursement Auditing with Evidence Verification and Local Correction

ReimburseAudit is a reimbursement-document auditing research and engineering project. It combines a shared multimodal document reader with three learned modules: policy-relation judgment, evidence-contribution verification, and local correction. The system checks document evidence, applies policy and amount constraints, and produces audit decisions, amount tables, trace records, and review requests.

## Architecture

1. **Document reading:** a Qwen3-VL-8B-Instruct backbone and a domain-adapted reader LoRA extract document facts and encode evidence.
2. **Policy-relation judgment:** a learned module evaluates policy relations and evidence roles.
3. **Evidence-contribution verification:** a second module examines how documents support or contradict an audit decision.
4. **Local correction:** a third module performs bounded correction before policy checks and exact amount calculations produce the final result.

The implementation uses a shared 512-dimensional audit representation, 256-dimensional internal networks, and eight role slots. Full details are in [the architecture guide](模型训练/02_实现工程/ARCHITECTURE.md).

## Repository layout

```text
ReimburseAudit/
├── README.md
├── 模型训练/
│   ├── 01_专利文档/          # Technical disclosures and implementation guide
│   ├── 02_实现工程/          # Source, configurations, tests, and environment specifications
│   ├── 03_评测结果/          # Evaluation summaries and verification records
│   └── ...                  # Retained review documents and historical source snapshots
└── backup/                  # Complete snapshot manifest and restoration instructions
```

## Setup and usage

See [environment installation](模型训练/02_实现工程/环境安装.md) and [training, inference, and evaluation commands](模型训练/02_实现工程/三模块快速运行.md). The engineering environment is documented for Linux, Python 3.12, and the NVIDIA PyTorch 25.08 environment; exact package versions are recorded in the project.

```bash
cd 模型训练/02_实现工程
python -m pip install -r requirements-runtime.lock.txt
```

Prepare the documented backbone, reader adapter, feature inputs, and trained checkpoints before running the pipeline. These large artifacts are not part of the lightweight Git checkout; consult the complete backup release for the archived working files.

## Recorded development evaluation

The delivery records report 12 jointly correct cases out of 23 fixed development workflows (52.17%), using both the audit flag and maximum compliant amount as the joint-correctness criterion. The recorded original Qwen and untuned GLM baselines each obtain 6/23 (26.09%). These figures describe the fixed development evaluation documented with this snapshot.

## Complete backup

The complete original directory, including data, trained weights, LoRA adapters, and historical runs, is backed up through split GitHub Release assets when uploaded. [backup/README.md](backup/README.md) records the inventory, checksums, exclusions, and restoration procedure. Source paths are preserved under `模型训练/`.
