from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import os,json,shutil,hashlib,re
R=Path('/mnt/public/zhouyibo/模型训练');E=R/'02_实现工程';W=Path('/mnt/public/zhouyibo/交付修订工作区_20261027_181327');state=json.loads((W/'session.json').read_text());A=Path(state['archive']);changes=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def replace(path,a,b):
 s=path.read_text();assert a in s,(str(path),a);path.write_text(s.replace(a,b));changes.append({'file':str(path.relative_to(R)),'before':a,'after':b})
# All original working figures are preserved byte-for-byte in the archive.
work_images={str(p.relative_to(R)):sha(p) for p in (E/'work').rglob('*') if p.is_file() and p.suffix.lower() in {'.png','.svg','.jpg','.jpeg'}}
raw=E/'data_raw/ReimburseBench';names=['patent_variants','patent_variants_shared_official_0822','patent_variants_v2']
D=E/'data/三模块交付_20261027';refs=[]
for p in [D/'dataset/reader_cases.jsonl',D/'reader_qa/train.jsonl',D/'reader_qa/validation.jsonl']:
 for line in p.read_text().splitlines():
  obj=json.loads(line)
  if 'evidence' in obj:refs.extend(x['raw_ref'] for x in obj['evidence'])
  elif 'raw_ref' in obj:refs.append(obj['raw_ref'])
assert not any(any(x.startswith(n+'/') for n in names) for x in refs)
moves=[E/'work',E/'data/patent_full',*(raw/n for n in names),raw/'.git']
for base,dirs,files in os.walk(E):
 if Path(base)==E:dirs[:]=[d for d in dirs if d!='work']
 for d in list(dirs):
  if d=='.git':dirs.remove(d)
  elif d in {'__pycache__','.pytest_cache','.ruff_cache'}:moves.append(Path(base)/d);dirs.remove(d)
archived=[]
for p in moves:
 if not p.exists():continue
 rel=p.relative_to(R);q=A/'归档文件'/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(p),q);archived.append(str(rel))
for rel,expected in work_images.items():assert sha(A/'归档文件'/rel)==expected
(raw/'SOURCE_REVISION').write_text(state['source_revision']+'\n')
replace(E/'data/patent_example/README.txt','工程示例：合成案例及标签，用于核对字段和运行接口。正式训练使用data/patent_full中的原始业务材料及核验后的条件级标签。','工程示例：本目录包含3个合成案例及配套标签，用于核对字段和运行接口。当前业务输入、监督和特征保存在data/三模块交付_20261027/，训练监督位于dataset/annotations.jsonl。')
replace(E/'data/packed/pack_report.md','当前参考目标均未批准用于三个新增神经部件的条件级监督。','本包保存原始整单参考目标；三个神经模块的条件级监督保存在 `data/三模块交付_20261027/dataset/annotations.jsonl`。')
replace(E/'data/packed/pack_report.md','## 数据边界','## 数据用途')
replace(E/'data/packed/pack_report.md','`reference_only` 表示可用于追溯和后续审核，不等于已批准的关系/引用训练标签。','`reference_only` 保存原始参考目标；条件关系与引用监督使用经核验的条件级标签及有效掩码。')
replace(E/'data/packed/pack_report.md','SFT读取问答尚未生成；需要先解析材料并人工核验金额、日期、人员等事实。','配套材料读取问答位于 `data/三模块交付_20261027/reader_qa/`，包含116条训练问答与4条验证问答，均绑定原始材料来源。')
for p in (E/'data/normalized_full').glob('*_parse_report.md'):
 replace(p,'取得了部分内容，但仍需读取图片或检查公式。','已取得结构化内容，关联图片读取或公式核验任务。')
 replace(p,'保留原图，不伪造 OCR 文本。','保留原图及视觉读取入口。')
replace(E/'data/splits/split_report.md','感知哈希只适合筛查相似图片，不能单独证明两张票据同源。','感知哈希筛选相似图片候选，同源关系结合原始票据与事件信息审核确认。')
p=D/'source_provenance/source_integrity.json';j=json.loads(p.read_text());j['dataset_run']='data/三模块交付_20261027';j['path_base']='02_实现工程';p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n')
replace(E/'code/scripts/pack_training_data.py','当前参考目标均未批准用于三个新增神经部件的条件级监督。','本包保存原始整单参考目标；条件级监督经原材料和政策审核后独立保存。')
replace(E/'code/scripts/prepare_patent_data.py','default=Path("data/patent_full")','default=Path("work/prepared_dataset")')
p=E/'code/scripts/audit_reimbursebench.py';s=p.read_text();a='def git_commit(repo_dir: Path) -> str | None:\n';assert a in s
b='''def git_commit(repo_dir: Path) -> str | None:
    recorded = repo_dir / "SOURCE_REVISION"
    if recorded.is_file():
        revision = recorded.read_text(encoding="utf-8").strip()
        if re.fullmatch(r"[0-9a-f]{40}", revision):
            return revision
    if not (repo_dir / ".git").exists():
        return None
'''
p.write_text(s.replace(a,b))
# The NVIDIA image supplies the locked torch build. Other packages get a separate install list.
locked=(E/'requirements.lock.txt').read_text().splitlines();extra=[s for s in locked if not s.startswith('torch==')]
(E/'requirements-runtime.lock.txt').write_text('# Install inside the documented NVIDIA PyTorch environment.\n'+'\n'.join(extra)+'\n')
install='''# 环境安装与核验

工程运行目录为 `02_实现工程/`。完整读取与训练环境采用Linux、Python 3.12及RTX5090，运行配置为 `configs/patent_delivery.yaml`。

## 准备项目对应的PyTorch环境

NVIDIA PyTorch 25.08镜像提供Python 3.12、CUDA 13.0和PyTorch源版本 `2.8.0a0+34c6371d24`，与 `environment.lock.json` 记录的NVIDIA构建相对应。版本依据见[NVIDIA 25.08官方说明](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-08.html)。宿主机先准备Docker、NVIDIA Container Toolkit，以及满足该镜像CUDA要求的NVIDIA驱动；驱动版本按官方说明链接的兼容表核对。

在本工程目录执行：

```bash
docker pull nvcr.io/nvidia/pytorch:25.08-py3
docker run --rm --gpus all --ipc=host -it \\
  --mount "type=bind,src=$PWD,dst=/workspace/project" \\
  --workdir /workspace/project \\
  nvcr.io/nvidia/pytorch:25.08-py3 bash
```

进入容器后，建立继承容器PyTorch的独立Python环境，并安装项目其余锁定依赖：

```bash
python -m venv --system-site-packages /opt/reimbursement-env
source /opt/reimbursement-env/bin/activate
python -m pip --isolated install -r requirements-runtime.lock.txt
export TMPDIR="$PWD/work/tmp"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$TMPDIR"
```

`requirements.lock.txt` 保存完整训练环境版本；`requirements-runtime.lock.txt` 为其配套安装清单，由容器提供PyTorch，其余版本逐项保持一致。虚拟环境安装按本项目版本执行，pip的isolated模式独立于容器的默认pip约束配置。

## 核验版本与设备

```bash
python -c 'import sys, torch; print(sys.version); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())'
python code/scripts/check_training_environment.py \\
  --config configs/patent_delivery.yaml --output-dir work/environment_check
```

版本核对目标：Python 3.12、PyTorch构建源版本 `2.8.0a0+34c6371d24`、CUDA 13.0。GPU读取与训练时，CUDA检查返回 `True`。包版本以 `environment.lock.json` 与 `requirements.lock.txt` 为准。

## 准备底座与运行

原始Qwen与GLM的基线预测、评分及设置保存在 `../03_评测结果/基线评测/`。当前交付模型权重位于 `models/报销审计_三模块/checkpoint_20261027_5217/`。

重建底座特征或继续材料读取训练时，将指定revision的Qwen底座放到 `models/Qwen3-VL-8B-Instruct/`，可使用以下入口：

```bash
python code/scripts/download_checkpoint.py \\
  --config configs/patent_delivery.yaml \\
  --output-dir models/Qwen3-VL-8B-Instruct
```

底座revision固定为 `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`。使用配套缓存特征进行CPU推理、评分及继续训练的完整命令见[三模块快速运行](三模块快速运行.md)。
'''
(E/'环境安装.md').write_text(install)
p=E/'README.md';s=p.read_text();anchor='工程已通过242项测试';s=s.replace(anchor,'环境准备与安装命令见[环境安装](环境安装.md)。其中PyTorch由指定NVIDIA镜像提供，其余依赖使用 `requirements-runtime.lock.txt` 安装。\n\n'+anchor);p.write_text(s)
p=E/'三模块快速运行.md';s=p.read_text().replace('下文的 `python` 应指向安装了 `requirements.lock.txt` 依赖的解释器。','先按[环境安装](环境安装.md)准备指定NVIDIA PyTorch环境和配套依赖，下文的 `python` 指向该解释器。');p.write_text(s)
p=next((R/'01_专利文档').glob('*.md'));s=p.read_text();anchor='### 3.2 当前参数';assert anchor in s;s=s.replace(anchor,'完整安装步骤见 `02_实现工程/环境安装.md`。PyTorch由NVIDIA PyTorch 25.08镜像提供，其他锁定依赖使用 `requirements-runtime.lock.txt` 安装；完整版本记录继续使用 `requirements.lock.txt` 和 `environment.lock.json`。\n\n'+anchor);p.write_text(s)
p=R/'交付说明.txt';s=p.read_text();s += '\n环境安装步骤：02_实现工程/环境安装.md\n其他锁定依赖安装清单：02_实现工程/requirements-runtime.lock.txt\n';p.write_text(s)
(E/'data/README.md').write_text('''# 数据目录说明

当前三模块训练与评测使用 `三模块交付_20261027/`：

| 内容 | 路径 | 数量 |
|---|---|---|
| 已读取业务输入 | `三模块交付_20261027/dataset/reader_cases.jsonl` | 67个输入 |
| 已核验条件监督 | `三模块交付_20261027/dataset/annotations.jsonl` | 198条 |
| 固定来源组划分 | `三模块交付_20261027/dataset/partitions.jsonl` | 配套输入划分 |
| 完整验证输入 | `三模块交付_20261027/validation_cases.jsonl` | 23例 |
| 材料读取问答 | `三模块交付_20261027/reader_qa/` | 训练116条、验证4条 |
| 真实底座特征与材料缓存 | `三模块交付_20261027/features/` | 与当前输入和LoRA绑定 |

`packed/` 和 `normalized_full/` 保存151个原始工作流的打包与解析结果；`splits/` 保存完整来源组划分；`labels_private/` 保存原始参考目标；`patent_schemas/` 保存当前接口Schema；`patent_example/` 保存3个合成接口示例。当前模块训练和推理按上表路径运行。

原始业务文件位于 `../data_raw/ReimburseBench/`。数据源版本保存于其 `SOURCE_REVISION` 文件及工程 `environment.lock.json`；各输入通过文件摘要绑定原始来源。
''')
result={'archive':str(A),'archived_paths':archived,'archived_work_images':len(work_images),'work_images_unchanged':True,'source_revision':state['source_revision'],'text_changes':changes,'word_pdf_changed':False,'formal_weights_changed':False}
(W/'applied_changes.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['text_changes','archived_paths']},ensure_ascii=False))
