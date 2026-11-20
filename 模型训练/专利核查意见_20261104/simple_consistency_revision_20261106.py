"""Small follow-up edits, preserving the existing Word revisions and equations."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import xml.dom.minidom as D
import zipfile

from tracked_revision_20261105 import W, M, el, plain, preview
from fix_revision_symbol_display_20261105 import paragraph_texts, visible_text

BASE = Path(__file__).resolve().parent
SOURCE = BASE.parent/'01_专利文档/11.3-一稿-一种基于审计过程数据的报销审计系统及方法.docx'
BACKUP = BASE/'六点简修前备份_20261106'/SOURCE.name
DRAFT = BASE/'周奕博_六点简修校验稿_20261106.docx'
DELIVERY = SOURCE.with_name('11.3-一稿-周奕博修订_一致性简修版_20261106.docx')
AUTHOR = '周奕博'
STAMP = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def elements(n):
    return [c for c in n.childNodes if c.nodeType == c.ELEMENT_NODE]


def accepted_map(p):
    text, locations = [], []

    def walk(node):
        if node.nodeType != node.ELEMENT_NODE:
            return
        if node.namespaceURI == W and node.localName == 'del':
            return
        if node.namespaceURI == M and node.localName in ('oMath', 'oMathPara'):
            accepted = node.cloneNode(True)
            for deleted in list(accepted.getElementsByTagNameNS(W, 'del')):
                deleted.parentNode.removeChild(deleted)
            value = visible_text(accepted)
            text.extend(value)
            locations.extend([None] * len(value))
            return
        if node.namespaceURI == W and node.localName == 't':
            value = plain(node)
            run = node.parentNode
            assert run.namespaceURI == W and run.localName == 'r'
            before = 0
            for t in run.getElementsByTagNameNS(W, 't'):
                if t is node:
                    break
                before += len(plain(t))
            for offset, letter in enumerate(value):
                text.append(letter)
                locations.append((run, before + offset))
            return
        if node.namespaceURI == W and node.localName in ('br', 'tab'):
            text.append('\n' if node.localName == 'br' else '\t')
            locations.append(None)
            return
        for child in elements(node):
            walk(child)

    walk(p)
    return ''.join(text), locations


class RevisionEditor:
    def __init__(self, doc):
        self.doc = doc
        self.ps = doc.getElementsByTagNameNS(W, 'body')[0].getElementsByTagNameNS(W, 'p')
        nodes = list(doc.getElementsByTagNameNS(W, 'ins')) + list(doc.getElementsByTagNameNS(W, 'del'))
        self.serial = max(int(n.getAttributeNS(W, 'id')) for n in nodes)
        self.log = []
        self.new_revision_ids = []

    def revision(self, tag):
        self.serial += 1
        self.new_revision_ids.append(self.serial)
        return el(self.doc, 'w', tag, W, id=self.serial, author=AUTHOR, date=STAMP)

    def fragment(self, original, value, deleted=False, extras=False):
        result = original.cloneNode(True)
        for c in list(result.childNodes):
            if c.nodeType != c.ELEMENT_NODE:
                result.removeChild(c)
            elif c.namespaceURI == W and c.localName in ('t', 'delText'):
                result.removeChild(c)
            elif not (c.namespaceURI == W and c.localName == 'rPr') and not extras:
                result.removeChild(c)
        t = el(self.doc, 'w', 'delText' if deleted else 't', W)
        t.setAttributeNS('http://www.w3.org/XML/1998/namespace', 'xml:space', 'preserve')
        t.appendChild(self.doc.createTextNode(value))
        result.appendChild(t)
        return result

    def replace(self, paragraph, old, new, category):
        p = self.ps[paragraph-1]
        text, mapping = accepted_map(p)
        start = text.find(old)
        assert start >= 0, (paragraph, old)
        assert text.find(old, start+1) < 0, ('ambiguous target', paragraph, old)
        selected = mapping[start:start+len(old)]
        assert selected and all(v is not None for v in selected), ('Would alter a native equation',paragraph,old)
        groups = []
        for run, offset in selected:
            if groups and groups[-1][0] is run:
                assert offset == groups[-1][2]
                groups[-1][2] += 1
            else:
                groups.append([run,offset,offset+1])
        inserted = False
        for run, lo, hi in groups:
            parent = run.parentNode
            assert parent.namespaceURI == W and parent.localName in ('p','ins'), parent.tagName
            original = ''.join(plain(t) for t in run.getElementsByTagNameNS(W,'t'))
            prefix, removed, suffix = original[:lo], original[lo:hi], original[hi:]
            extras_available = True

            def append_fragment(value, deleted=False):
                nonlocal extras_available
                node = self.fragment(run,value,deleted,extras_available)
                extras_available = False
                return node

            if prefix:
                parent.insertBefore(append_fragment(prefix),run)
            if parent.localName == 'ins':
                # Editing one's existing insertion retains that insertion as pending.
                assert parent.getAttributeNS(W,'author') == AUTHOR
                if not inserted:
                    parent.insertBefore(append_fragment(new),run)
                    inserted = True
                elif extras_available:
                    # Keep cached page-break markup even when the inserted text vanishes.
                    nontext = [n for n in elements(run) if n.localName not in ('rPr','t')]
                    if nontext:
                        parent.insertBefore(append_fragment(''),run)
            else:
                deletion = self.revision('del')
                deletion.appendChild(append_fragment(removed,True))
                parent.insertBefore(deletion,run)
                if not inserted:
                    insertion = self.revision('ins')
                    insertion.appendChild(self.fragment(run,new))
                    parent.insertBefore(insertion,run)
                    inserted = True
            if suffix:
                parent.insertBefore(append_fragment(suffix),run)
            parent.removeChild(run)
        actual,_ = accepted_map(p)
        assert actual == text[:start]+new+text[start+len(old):]
        self.log.append({'paragraph':paragraph,'category':category,'before':old,'after':new})


def apply(e):
    # The six requested corrections.
    e.replace(231,'验证材料读取、条件判断、文件引用、自然错误净修正及误改率，人工扰动恢复率单独报告。',
              '可采用材料读取准确率、条件判断准确率、引用准确率、自然错误净修正率及误改率等指标，评价系统各环节性能。','1_评价方法与结果区分')
    e.replace(221,'整单答案监督沿用既有判定、最高金额和问题说明',
              '既有整单参考答案保留判定、最高金额和问题说明，用于条件级标签构建及整单结果评测','2_整单答案用途')
    e.replace(240,'为了说明上述系统及方法的有效性和可行性，下面结合一个具体案例展开说明。',
              '以下通过住宿费报销示例，说明系统的执行步骤、信息缺口处理及金额划分规则。','3_流程示例')
    e.replace(53,'而不扰动其他已正确的审计条件',
              '，其他未触发校正的审计条件保持原有表示及结果','4_旁路保证范围')
    e.replace(118,'局部校正不扰动已正确条件','局部校正仅作用于触发条件','4_旁路保证范围')
    e.replace(118,'已正确的其他条件不被重新生成改变',
              '未触发校正的其他条件沿用原有表示及结果','4_旁路保证范围')
    e.replace(118,'由此避免一处错误导致整份审计结果重新生成',
              '由此将重新计算限制在触发校正的条件范围内','4_旁路保证范围')
    e.replace(245,'其他无错误条件保持原结果','其他未触发校正的条件保持原结果','4_旁路保证范围')
    e.replace(117,'使引用判断不仅依据语义相关性，还依据该证据对条件结论的实际影响',
              '使引用判断综合采用语义关系、材料贡献与必要信息覆盖信息','5_贡献核验机制')
    e.replace(117,'等效证据可相互替代时，材料贡献为零不被自动认定为无效引用',
              '等效证据通过信息覆盖与来源核验择一引用','5_贡献核验机制')
    e.replace(117,'从而在保留等效引用灵活性的同时，避免仅凭引用列表误判证据充分性',
              '在保留等效引用灵活性的同时，为证据充分性判断增加材料贡献和必要信息覆盖依据','5_贡献核验机制')
    e.replace(224,'第三阶段固定',
              '第三阶段训练局部校正子模块，可使用自然错误及经筛选的扰动错误样本。本次实施例采用经核验的自然错误反馈样本。可选的扰动样本构造方式如下：固定','6_通用方案与本次实施')
    e.replace(104,'固定报销申请数据',
              '局部校正训练使用自然错误样本，并可采用扰动错误样本作为补充；构造扰动样本时，固定报销申请数据','6_通用方案与本次实施')
    e.replace(104,'以恢复所述原正确事实校验数据中的关系、引用和状态为目标，训练局部校正参数',
              '对筛选通过的扰动错误样本，以恢复所述原正确事实校验数据中的关系、引用和状态为目标训练局部校正参数','6_通用方案与本次实施')
    e.replace(227,'自然错误及正确样本同时用于训练和验证纠错与误改情况',
              '可采用自然错误和正确条件分别评价纠错与误改情况','6_通用方案与本次实施')
    e.replace(227,'本实施例取0.03','该可选分支取0.03','6_通用方案与本次实施')
    e.replace(227,'本实施例取','该分支取','6_通用方案与本次实施')
    e.replace(245,'从完整训练案例删除入住文件后',
              '在该示例中，若删除入住文件','3_流程示例')
    e.replace(245,'内部表示扰动则保持','采用可选扰动分支时，保持','6_通用方案与本次实施')

    # Small positive rewrites; preserve masks, failure handling and mathematical rules.
    e.replace(162,'不能丢弃必要信息','拆分后保留全部必要信息','表述简化')
    e.replace(94,'错误分量归零','当前条件的错误向量为零','停止条件一致')
    e.replace(205,'错误分量归零','错误向量为零','停止条件一致')
    e.replace(205,'停止后仍存在异常、必要信息未决或支持与冲突同时达标时进入人工复核；连续两轮不变触发停止，停止后存在的残余异常继续转入人工复核。',
              '停止后存在残余异常、必要信息未决或支持与冲突同时达标时，转入人工复核。','表述简化')
    e.replace(205,'开放文本条件按状态、依据和阈值判定是否可处理，不因不能程序比较而一律拒绝。',
              '开放文本条件结合语义状态、证据依据和阈值进入相应处理分支。','表述简化')
    e.replace(214,'缺材料按政策区分允许补件的待复核与明确不予报销的当前申请，不能统一判为拒绝。',
              '缺材料时，按政策分别进入补件复核或拒绝当前申请的处理分支。','表述简化')
    e.replace(216,'无须另行训练三个大语言模型','三个神经模块在共享底座之上分别训练','表述简化')
    e.replace(220,'答案未提到某条件不等于该条件成立','各条件标签依据原材料和政策逐项核验','表述简化')
    e.replace(220,'未标注替代材料不直接作为负例','替代材料经等效关系与信息覆盖核验后确定监督标签','表述简化')
    e.replace(222,'引用头预测文件级引用，文本片段选择由读取模型与程序匹配辅助，不要求新增模块生成像素坐标或长篇审计答案。',
              '引用头负责文件级引用，文本片段选择由读取模型与程序匹配辅助。','表述简化')
    e.replace(222,'底座可冻结，材料读取有经核验问答时再进行可选监督微调（SFT）与LoRA；不以完整字段或位置标注作为必要前提。',
              '底座可冻结，经核验的读取问答可用于材料读取的监督微调（SFT）与LoRA适配。','表述简化')
    e.replace(223,'不只修改模型读取值而保留相矛盾原件。',
              '事实变化同步更新副本材料、读取操作数及依赖关系。','表述简化')
    e.replace(230,'不将贡献为零硬设为负例','引用标签结合经核验的依据与覆盖关系确定','表述简化')
    e.replace(230,'未触发条件采用旁路，其表示保持是实现约束，无需恒零保持损失。',
              '未触发校正的条件通过旁路保持当前表示。','表述简化')
    e.replace(270,'本发明使政策运算约束语义判断、证据引用可核验、局部错误可校正且金额守恒、路径可追溯。',
              '本发明将政策运算融入关系判断，通过材料贡献核验证据引用，按条件触发局部表示更新，并以确定性程序完成金额划分及路径记录。','效果与机制一致')


def check(before, after, package, new_ids):
    a = preview(after,True)
    texts = paragraph_texts(a)
    combined='\n'.join(texts)
    banned=('人工扰动恢复率单独报告','整单答案监督','为了说明上述系统及方法的有效性和可行性',
            '不扰动其他已正确','不扰动已正确条件','已正确的其他条件','其他无错误条件',
            '避免仅凭引用列表误判','60.9%','第9节','如图4所示','更新错误分量为零的条件')
    assert not [s for s in banned if s in combined]
    assert '本次实施例采用经核验的自然错误反馈样本' in combined
    assert '可选的扰动样本构造方式如下' in combined
    assert '原判断正确' in texts[223]  # This is a supervised training filter.
    assert texts[262] == '52.17%（12/23）'
    assert texts[263] == texts[264] == '26.09%（6/23）'
    assert '固定调参验证集' in texts[265]
    for p in (161,169,176,182,204,214,215,227):
        assert texts[p-1]
        assert '其中为' not in texts[p-1]
        assert '记为，' not in texts[p-1]
    # No mathematical edit is needed for this follow-up.
    old_math=before.getElementsByTagNameNS(M,'oMath')
    new_math=after.getElementsByTagNameNS(M,'oMath')
    assert [n.toxml() for n in old_math] == [n.toxml() for n in new_math]
    assert paragraph_texts(preview(before,False)) == paragraph_texts(preview(after,False))
    revs=list(after.getElementsByTagNameNS(W,'ins'))+list(after.getElementsByTagNameNS(W,'del'))
    assert len({n.getAttributeNS(W,'id') for n in revs}) == len(revs)
    assert all(n.getAttributeNS(W,'author') == AUTHOR for n in revs)
    assert not any(n.getElementsByTagNameNS(M,'oMath') for n in revs)
    core=(128,159,160,167,168,170,174,175,178,199,201,202,203,210,211,212,213,225,226,229)
    ps=a.getElementsByTagNameNS(W,'body')[0].getElementsByTagNameNS(W,'p')
    assert all(ps[i-1].getElementsByTagNameNS(M,'oMath') for i in core)
    return texts,{'six_requested_issues_resolved':True,'training_scope_consistent':True,
                  'untouched_conditions_not_claimed_correct':True,'metrics_described_as_optional_evaluation':True,
                  'evaluation_values_preserved':True,'all_native_equations_unchanged':len(old_math),
                  'reject_all_preserves_previous_original_reading':True,'revision_author':AUTHOR,
                  'new_revision_records':len(new_ids),'revision_ids_unique':True,
                  'no_hidden_formula_revision_nesting':True}


def main():
    current_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if BACKUP.exists():
        assert hashlib.sha256(BACKUP.read_bytes()).hexdigest() == current_hash
    else:
        BACKUP.parent.mkdir(parents=True)
        shutil.copy2(SOURCE,BACKUP)
    with zipfile.ZipFile(BACKUP) as z:
        package={n:z.read(n) for n in z.namelist()}
        info=z.infolist()
    before=D.parseString(package['word/document.xml'])
    after=D.parseString(package['word/document.xml'])
    editor=RevisionEditor(after)
    apply(editor)
    texts,checks=check(before,after,package,editor.new_revision_ids)
    updated=after.toxml(encoding='UTF-8')
    with zipfile.ZipFile(DRAFT,'w') as z:
        for entry in info:
            z.writestr(entry,updated if entry.filename=='word/document.xml' else package[entry.filename])
    with zipfile.ZipFile(DRAFT) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if name.endswith('.xml') or name.endswith('.rels'):
                D.parseString(z.read(name))
            if name!='word/document.xml':
                assert z.read(name)==package[name]
    report={'source':str(SOURCE),'backup':str(BACKUP),'source_sha256_before':current_hash,
            'draft':str(DRAFT),'delivery':str(DELIVERY),'checks':checks,
            'edits':editor.log,'source_updated':False}
    (BASE/'六点简修与一致性核验_20261106.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (BASE/'六点简修后_全文校对文本_20261106.txt').write_text('\n\n'.join(f'[{i+1:03}] {s}' for i,s in enumerate(texts)))
    print(json.dumps({'draft':str(DRAFT),'edits':len(editor.log),'checks':checks},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
