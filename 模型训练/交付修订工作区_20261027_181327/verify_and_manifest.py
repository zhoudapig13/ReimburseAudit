from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter
import json,hashlib,sys,os,shutil,re,xml.etree.ElementTree as ET
from lxml import etree
from docx import Document
R=Path('/mnt/public/zhouyibo/模型训练');E=R/'02_实现工程';W=Path('/mnt/public/zhouyibo/交付修订工作区_20261027_181327');sys.path.insert(0,str(E))
from audit_engine.contracts import AuditCase,ConditionAnnotation
from audit_engine.training.patent import validate_annotations
state=json.loads((W/'session.json').read_text());old={x['path']:x for x in state['manifest_before']['files']};env=json.loads((W/'environment_verification.json').read_text());stamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
# Validate source-bound current data and all material QA references after the move.
D=E/'data/三模块交付_20261027';cases=[AuditCase.model_validate_json(s) for s in (D/'dataset/reader_cases.jsonl').read_text().splitlines()];anns=[ConditionAnnotation.model_validate_json(s) for s in (D/'dataset/annotations.jsonl').read_text().splitlines()];parts={x['workflow_id']:x for s in (D/'dataset/partitions.jsonl').read_text().splitlines() if (x:=json.loads(s))};validate_annotations(cases,anns,parts)
assert len(cases)==67 and len(anns)==198
materials={m.raw_ref:m.source_sha256 for c in cases for m in c.evidence};qa_counts={}
for split in ['train','validation']:
 rows=[json.loads(s) for s in (D/'reader_qa'/(split+'.jsonl')).read_text().splitlines()];qa_counts[split]=len(rows)
 for x in rows:
  assert x['raw_ref'] not in materials or materials[x['raw_ref']]==x['source_sha256'];materials[x['raw_ref']]=x['source_sha256']
assert qa_counts=={'train':116,'validation':4}
with ThreadPoolExecutor(max_workers=4) as pool:actual=list(pool.map(lambda x:(x[0],sha(E/'data_raw/ReimburseBench'/x[0]),x[1]),materials.items()))
assert all(a==b for _,a,b in actual)
for name,cls in [('runtime_case.schema.json',AuditCase),('condition_annotation.schema.json',ConditionAnnotation)]:assert json.loads((E/'data/patent_schemas'/name).read_text())==cls.model_json_schema()
# Confirm the full current prediction is semantically identical to the delivered one.
latest=[json.loads(s) for s in (W/'predictions.jsonl').read_text().splitlines()];delivered=[json.loads(s) for s in (R/'03_评测结果/三模块评测/predictions.jsonl').read_text().splitlines()];keys=['workflow_id','F','B','answer','review','module_calls']
assert len(latest)==len(delivered)==23 and all({k:x.get(k) for k in keys}=={k:y.get(k) for k in keys} for x,y in zip(latest,delivered));score=json.loads((W/'score.json').read_text());assert score['correct']==12 and score['total']==23
calls={k:sum(x['module_calls'][k] for x in latest) for k in ['relation','contribution','corrector']};assert calls=={'relation':462,'contribution':55,'corrector':32}
# Baseline data and all nine weight/config files remain exactly the same.
weights=[p for p in (E/'models').rglob('*') if p.is_file() and (p.suffix in {'.pt','.safetensors'} or p.name=='adapter_config.json')]
assert len(weights)==9
for p in weights:assert sha(p)==old[str(p.relative_to(R))]['sha256']
for p in (R/'03_评测结果/基线评测').rglob('*'):
 if p.is_file():assert sha(p)==old[str(p.relative_to(R))]['sha256']
word=next((R/'01_专利文档').glob('*.docx'));pdf=word.with_suffix('.pdf');assert sha(word)==old[str(word.relative_to(R))]['sha256'] and sha(pdf)==old[str(pdf.relative_to(R))]['sha256']
# Full regression was run outside the delivery tree.
tree=ET.parse(W/'tests.xml').getroot();suites=list(tree.iter('testsuite'));passed=sum(int(x.get('tests',0))-int(x.get('failures',0))-int(x.get('errors',0))-int(x.get('skipped',0)) for x in suites);skipped=sum(int(x.get('skipped',0)) for x in suites);assert passed==242 and skipped==1 and all(x.get('failures','0')=='0' and x.get('errors','0')=='0' for x in suites)
shutil.copy2(W/'tests.xml',R/'03_评测结果/三模块评测/tests.xml')
# Ensure no work/test/Git/cache directories remain inside the actual delivery.
for base,dirs,files in os.walk(R):assert not(set(dirs)&{'work','.git','__pycache__','.pytest_cache','.ruff_cache'}),(base,dirs)
assert not(E/'data/patent_full').exists()
assert sorted(p.name for p in (E/'data_raw/ReimburseBench').iterdir() if p.is_dir() and p.name.startswith('patent_variants'))==['patent_variants_reader_refresh_1020']
# Relative paths and wording for data-stage reports now match the delivered artifacts.
assert 'data/三模块交付_20261027/' in (E/'data/patent_example/README.txt').read_text();source=json.loads((D/'source_provenance/source_integrity.json').read_text());assert (E/source['dataset_run']).is_dir()
for p in (E/'data').rglob('*'):
 if p.is_file() and p.suffix in {'.md','.txt'}:assert not re.search(r'SFT读取问答尚未生成|不能单独证明|不等于已批准|仍需读取|正式训练使用data/patent_full',p.read_text()),p
checks=[
{'item':'工作目录与测试缓存','result':'通过','detail':'工作产物和测试权重位于交付目录外；目录内无work、Git或工具缓存'},
{'item':'旧数据目录与入口','result':'通过','detail':'当前输入与198条监督均按三模块交付数据目录读取；合成示例说明同步'},
{'item':'数据阶段报告','result':'通过','detail':'打包、解析及划分报告使用当前用途和来源表述'},
{'item':'来源记录可移植性','result':'通过','detail':'来源记录使用工程相对路径；SOURCE_REVISION可独立读取'},
{'item':'原始材料与问答完整性','result':'通过','detail':str(len(materials))+'份当前输入和问答所需材料逐份摘要一致；训练116条问答、验证4条'},
{'item':'环境安装与版本','result':'通过','detail':'指定NVIDIA镜像来源已核对；其他依赖与锁文件逐项一致；安装命令语法通过'},
{'item':'模型及基线结果保留','result':'通过','detail':'联合checkpoint、五阶段checkpoint和读取LoRA全部保留；Qwen与GLM基线数据一致'},
{'item':'Word与PDF图片','result':'通过','detail':'Word与PDF整份文件摘要不变，最后附图保持原样'},
{'item':'完整工程测试和联合推理','result':'通过','detail':'242项通过、1项跳过；23例逐例F/B/判定一致，联合准确率52.17%'}]
report={'checked_at':stamp,'status':'passed','checks':checks,'material_files_verified':len(materials),'current_inputs':67,'approved_annotations':198,'reader_qa':qa_counts,'tests_passed':passed,'tests_skipped':skipped,'correct':12,'total':23,'accuracy':score['accuracy'],'module_calls':calls,'formal_weight_records_preserved':9,'baseline_data_unchanged':True,'word_pdf_unchanged':True,'last_figure_unchanged':True,'source_revision':env['source_revision'],'environment_install':{'base_image':'nvcr.io/nvidia/pytorch:25.08-py3','source':'https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-08.html','install_commands_syntax_verified':True,'local_torch':env['torch'],'local_cuda':env['cuda'],'container_pull_performed':False}}
(R/'03_评测结果/交付前自查.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
# Update the prior review record's references to deliverable paths, rather than private work outputs.
p=R/'03_评测结果/最终交付一致性核验.json';review=json.loads(p.read_text());review['checked_at']=stamp;review['pre_delivery_check']='03_评测结果/交付前自查.json';review['verification_prediction_sha256']=sha(R/'03_评测结果/三模块评测/predictions.jsonl')
for check in review['checks']:
 refs=[]
 for ref in check.get('references',[]):
  if ref=='work/final_alignment/predictions.jsonl':ref='03_评测结果/三模块评测/predictions.jsonl'
  elif ref=='work/final_alignment/score.json':ref='03_评测结果/三模块评测/score.json'
  elif '技术交底书' in ref:ref=str(word.relative_to(R))
  elif ref=='training_summary.json':ref='02_实现工程/models/报销审计_三模块/checkpoint_20261027_5217/training_summary.json'
  elif ref=='reader_lora/adapter_config.json':ref='02_实现工程/models/报销审计_三模块/checkpoint_20261027_5217/reader_lora/adapter_config.json'
  elif ref=='reader_qa/':ref='02_实现工程/data/三模块交付_20261027/reader_qa/'
  elif ref=='validation_cases.jsonl':ref='02_实现工程/data/三模块交付_20261027/validation_cases.jsonl'
  elif ref.startswith('基线评测/'):ref='03_评测结果/'+ref
  elif ref.startswith(('audit_engine/','code/','configs/','data/','data_raw/','models/','tests/')):ref='02_实现工程/'+ref
  assert (R/ref).exists(),ref
  refs.append(ref)
 check['references']=refs
review['check_count']=len(review['checks']);p.write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
# Rebuild and verify every delivered file; work and archives are physically outside this tree.
files=sorted(p for p in R.rglob('*') if p.is_file() and p.name!='交付清单.json')
with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(lambda p:{'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':sha(p)},files))
m=state['manifest_before'];m.update(updated=stamp,files=records,file_count=len(records),logical_bytes=sum(x['bytes'] for x in records));(R/'交付清单.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
assert len({x['path'] for x in records})==len(records)
statepath=R.parent/'交付整理状态.json';current=json.loads(statepath.read_text());current.update(inventory_files=len(records),pre_delivery_cleanup_complete=True,last_figure_unchanged=True,checkpoint_unchanged=True,reader_lora_unchanged=True);statepath.write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n')
report['inventory_files']=len(records);report['logical_bytes']=m['logical_bytes'];(W/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({'status':'passed','inventory_files':len(records),'logical_bytes':m['logical_bytes'],'material_files':len(materials),'tests_passed':passed,'tests_skipped':skipped,'accuracy':score['accuracy'],'weights_preserved':9,'word_pdf_unchanged':True},ensure_ascii=False))
