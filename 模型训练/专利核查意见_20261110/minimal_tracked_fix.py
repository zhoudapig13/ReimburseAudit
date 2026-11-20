"""Minimal final-draft corrections; keep all prior pending revisions intact."""
from pathlib import Path
import sys, json, hashlib, shutil, zipfile, os
import xml.dom.minidom as D

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent/'专利核查意见_20261104'))
from simple_consistency_revision_20261106 import RevisionEditor, accepted_map, elements, W, M, el, plain, preview

SOURCE = BASE.parent/'01_专利文档/11.9-二稿-一种基于审计过程数据的报销审计系统及方法(1).docx'
DRAFT = BASE/'11.9二稿_周奕博最小修订_校验稿.docx'
BACKUP = BASE/'源文件修改前备份'/SOURCE.name


def main():
    original = SOURCE.read_bytes()
    sha = hashlib.sha256(original).hexdigest()
    assert sha == 'baabb28b317046f9c07f7012e998aba2f88992fcd7f0790e544ab28e8fe0cce0'
    BACKUP.parent.mkdir(exist_ok=True)
    if BACKUP.exists():
        assert BACKUP.read_bytes() == original
    else:
        shutil.copy2(SOURCE, BACKUP)
    with zipfile.ZipFile(SOURCE) as z:
        infos = z.infolist()
        package = {x.filename:z.read(x.filename) for x in infos}
    before = D.parseString(package['word/document.xml'])
    doc = D.parseString(package['word/document.xml'])
    e = RevisionEditor(doc)
    ps = e.ps
    changed_math = []

    def replace(i, old, new, category):
        e.replace(i, old, new, category)

    def math_revision(run, value, i, label):
        parent = run.parentNode
        old = plain(run.getElementsByTagNameNS(M,'t')[0])
        deletion = e.revision('del')
        deletion.appendChild(run.cloneNode(True))
        insertion = e.revision('ins')
        added = run.cloneNode(True)
        added.getElementsByTagNameNS(M,'t')[0].firstChild.data = value
        insertion.appendChild(added)
        parent.insertBefore(deletion, run)
        parent.insertBefore(insertion, run)
        parent.removeChild(run)
        changed_math.append({'paragraph':i,'before':old,'after':value,'category':label})

    def mrun(value):
        r=el(doc,'m','r',M)
        rp=el(doc,'w','rPr',W)
        fonts=el(doc,'w','rFonts',W,ascii='Cambria Math',hAnsi='Cambria Math',eastAsia='宋体')
        rp.appendChild(fonts);r.appendChild(rp)
        t=el(doc,'m','t',M);t.appendChild(doc.createTextNode(value));r.appendChild(t)
        return r

    def inserted_math(parent, node, ref=None):
        rev=e.revision('ins');rev.appendChild(node)
        if ref:parent.insertBefore(rev,ref)
        else:parent.appendChild(rev)

    def new_indexed(base, sub):
        s=el(doc,'m','sSub',M)
        b=el(doc,'m','e',M);a=el(doc,'m','sub',M)
        inserted_math(b,mrun(base));inserted_math(a,mrun(sub))
        s.appendChild(b);s.appendChild(a)
        return s

    # Claims and corresponding general description: preserve the original structure.
    for i in (31,98):
        replace(i,'适用上限已确认时，取分项申请金额与适用上限中的较小值作为截断金额',
                '适用上限及材料支持金额已确认时，取分项申请金额、适用上限和材料支持金额中的最小值作为截断金额','金额约束')
        replace(i,'政策明确不设上限时，截断金额取分项申请金额',
                '政策明确不设上限时，截断金额取分项申请金额与已确认材料支持金额中的较小值','金额约束')
        replace(i,'适用上限因必要信息未决而待确认时，计算中暂取分项申请金额，并将相应费用单元划入待复核集合',
                '适用上限或材料支持金额因必要信息未决而待确认时，计算中按已确认约束截断分项申请金额；已有明确拒绝依据的费用单元保持不可报销状态，其余待确认单元划入待复核集合','拒绝优先级')

    replace(81,'政策条件所需的全部必要信息角色是否均已从证据中取得操作数',
            '政策条件所需的全部必要信息角色是否均已从证据中取得操作数，且执行条件所需的类型、币种、日期解释及规则适用信息是否已确认','缺口定义')
    replace(81,'必要信息缺口标记在必要信息尚未取得时取一，取得时取零',
            '必要信息缺口标记在必要信息尚未取得或上述执行信息未确认时取一，均已取得并确认时取零','缺口定义')
    replace(81,'区分材料缺失和模型尚未读出信息','区分材料缺失、读取失败和解释歧义','缺口定义')

    # V is already the defined error-vector symbol; change only the missed v tokens.
    for i, expected in ((200,1),(202,1),(203,1),(204,2),(232,1)):
        ts=[t for t in ps[i-1].getElementsByTagNameNS(M,'t') if plain(t)=='v']
        assert len(ts)==expected,(i,len(ts))
        for t in ts:math_revision(t.parentNode,'V',i,'错误向量符号统一')

    replace(207,'为按已确认适用上限截断后的金额','为按已确认适用上限及材料支持金额截断后的金额','金额约束')
    replace(208,'并令','，将经核验材料支持的金额记为M_i，并令','金额约束')
    # Use ordinary Word text with a real subscript for the new inline definition.
    candidates=[t for t in ps[207].getElementsByTagNameNS(W,'t') if '记为M_i' in plain(t)]
    assert len(candidates)==1
    t=candidates[0];r=t.parentNode;parent=r.parentNode
    assert parent.localName=='ins'
    pre,post=plain(t).split('M_i')
    for text,sub in ((pre,False),('M',False),('i',True),(post,False)):
        if not text:continue
        nr=e.fragment(r,text)
        if text in ('M','i'):
            rp=nr.getElementsByTagNameNS(W,'rPr')[0]
            fonts=rp.getElementsByTagNameNS(W,'rFonts')[0]
            fonts.setAttributeNS(W,'w:ascii','Cambria Math');fonts.setAttributeNS(W,'w:hAnsi','Cambria Math')
            if sub:rp.appendChild(el(doc,'w','vertAlign',W,val='subscript'))
        parent.insertBefore(nr,r)
    parent.removeChild(r)

    maths=list(ps[207].getElementsByTagNameNS(M,'oMath'))
    first=next(m for m in maths if accepted_map(m)[0]=='xi=min(ai,Li)')
    closing=elements(first)[-1]
    assert plain(closing.getElementsByTagNameNS(M,'t')[0])==')'
    inserted_math(first,mrun(','),closing)
    first.insertBefore(new_indexed('M','i'),closing)
    changed_math.append({'paragraph':208,'before':'x_i=min(a_i,L_i)','after':'x_i=min(a_i,L_i,M_i)','category':'金额约束'})

    replace(208,'适用上限因必要信息未决而待确认时，暂取',
            '适用上限或材料支持金额因必要信息未决而待确认时，待确认项以正无穷作计算占位，仍按','拒绝优先级')
    second=next(m for m in maths if accepted_map(m)[0]=='xi=ai')
    a_node=elements(second)[-1]
    inserted_math(second,mrun('min('),a_node)
    inserted_math(second,mrun(','));second.appendChild(new_indexed('L','i'))
    inserted_math(second,mrun(','));second.appendChild(new_indexed('M','i'));inserted_math(second,mrun(')'))
    changed_math.append({'paragraph':208,'before':'x_i=a_i','after':'x_i=min(a_i,L_i,M_i)','category':'已确认约束保留'})
    replace(208,'并将相应费用单元划入','计算截断金额，并将无明确拒绝依据的费用单元划入','拒绝优先级')
    replace(208,'。各分支采用上述明确取值','；已明确拒绝的单元仍归入不可报销集合。各分支采用上述明确取值','拒绝优先级')

    # One short definition resolves the S/C shorthand without rewriting the formula.
    replace(230,'依次对应满足、不满足、证据不足与事实矛盾，非负且和为',
            '其中S、C分别为当前条件的支持强度与冲突强度的简写；四个分量依次对应满足、不满足、证据不足与事实矛盾，非负且和为','概率符号定义')

    # Validate reversibility, scope, old revision history and mathematical visibility.
    old_ps=before.getElementsByTagNameNS(W,'body')[0].getElementsByTagNameNS(W,'p')
    old_revs=list(before.getElementsByTagNameNS(W,'ins'))+list(before.getElementsByTagNameNS(W,'del'))
    new_revs=list(doc.getElementsByTagNameNS(W,'ins'))+list(doc.getElementsByTagNameNS(W,'del'))
    by_id={r.getAttributeNS(W,'id'):r for r in new_revs}
    for r in old_revs:assert by_id[r.getAttributeNS(W,'id')].toxml()==r.toxml()
    assert len(by_id)==len(new_revs)
    assert all(by_id[str(i)].getAttributeNS(W,'author')=='周奕博' for i in e.new_revision_ids)
    assert not any(r.getElementsByTagNameNS(M,'oMath') for r in new_revs)
    rejected_old=preview(before,False);rejected_new=preview(doc,False)
    assert [accepted_map(p)[0] for p in rejected_old.getElementsByTagNameNS(W,'body')[0].getElementsByTagNameNS(W,'p')]==[accepted_map(p)[0] for p in rejected_new.getElementsByTagNameNS(W,'body')[0].getElementsByTagNameNS(W,'p')]
    accepted=preview(doc,True)
    accepted_ps=accepted.getElementsByTagNameNS(W,'body')[0].getElementsByTagNameNS(W,'p')
    texts=[accepted_map(p)[0] for p in accepted_ps]
    allowed={31,81,98,200,202,203,204,207,208,230,232}
    changed=[i for i,(a,b) in enumerate(zip(old_ps,ps),1) if a.toxml()!=b.toxml()]
    assert set(changed)==allowed,changed
    assert len(doc.getElementsByTagNameNS(M,'oMath'))==len(before.getElementsByTagNameNS(M,'oMath'))==355
    assert texts[207].count('xi=min(ai,Li,Mi)')==2,texts[207]
    assert '记为Mi' in texts[207]
    assert all(not any(plain(t)=='v' for t in accepted_ps[i-1].getElementsByTagNameNS(M,'t')) for i in (200,202,203,204,232))
    assert all(texts[i-1]==accepted_map(old_ps[i-1])[0] for i in (153,156,241,244,262,263,264,265))

    settings=D.parseString(package['word/settings.xml'])
    if not settings.getElementsByTagNameNS(W,'trackRevisions'):
        settings.documentElement.appendChild(el(settings,'w','trackRevisions',W))
    updates={'word/document.xml':doc.toxml(encoding='UTF-8'),'word/settings.xml':settings.toxml(encoding='UTF-8')}
    with zipfile.ZipFile(DRAFT,'w') as z:
        for info in infos:z.writestr(info,updates.get(info.filename,package[info.filename]))
    with zipfile.ZipFile(DRAFT) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if name.endswith(('.xml','.rels')):D.parseString(z.read(name))
            if name not in updates:assert z.read(name)==package[name]
    (BASE/'修改后正文_校对.txt').write_text('\n\n'.join(f'[{i:03}] {s}' for i,s in enumerate(texts,1)))
    report={'source':str(SOURCE),'backup':str(BACKUP),'draft':str(DRAFT),'sha_before':sha,
            'modified_paragraphs':changed,'previous_revisions_preserved':len(old_revs),
            'new_revision_records':len(e.new_revision_ids),'native_math_objects_preserved':355,
            'reject_all_restores_original_reading':True,'other_parts_including_images_unchanged':True,
            'track_revisions_enabled':True,'author':'周奕博','text_edits':e.log,'math_edits':changed_math,
            'source_updated':False}
    (BASE/'最小修订核验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('text_edits','math_edits')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
