"""Replace the broader edit with six letter corrections from the original backup."""
from pathlib import Path
from datetime import datetime, timezone
import sys, json, hashlib, shutil, os, zipfile
import xml.dom.minidom as D

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'专利核查意见_20261104'))
from simple_consistency_revision_20261106 import RevisionEditor, W, M, plain, accepted_map, preview


def main():
    prior=json.loads((BASE/'最小修订核验.json').read_text())
    source=Path(prior['source']);backup=Path(prior['backup'])
    current_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    assert current_sha==prior['sha_after'],'Source changed since the previous edit.'
    assert hashlib.sha256(backup.read_bytes()).hexdigest()==prior['sha_before']
    with zipfile.ZipFile(backup) as z:
        infos=z.infolist();package={i.filename:z.read(i.filename) for i in infos}
    before=D.parseString(package['word/document.xml'])
    doc=D.parseString(package['word/document.xml'])
    editor=RevisionEditor(doc)
    edits=[]
    for i,expected in ((200,1),(202,1),(203,1),(204,2),(232,1)):
        p=editor.ps[i-1]
        ts=[t for t in p.getElementsByTagNameNS(M,'t') if plain(t)=='v']
        assert len(ts)==expected
        for t in ts:
            run=t.parentNode;parent=run.parentNode
            deletion=editor.revision('del');deletion.appendChild(run.cloneNode(True))
            insertion=editor.revision('ins');new=run.cloneNode(True)
            new.getElementsByTagNameNS(M,'t')[0].firstChild.data='V'
            insertion.appendChild(new)
            parent.insertBefore(deletion,run);parent.insertBefore(insertion,run);parent.removeChild(run)
            edits.append({'paragraph':i,'before':'v','after':'V'})

    def texts(document):
        ps=document.getElementsByTagNameNS(W,'body')[0].getElementsByTagNameNS(W,'p')
        return [accepted_map(p)[0] for p in ps]

    original_text=texts(preview(before,True));final_text=texts(preview(doc,True))
    assert len(edits)==6
    for i,(old,new) in enumerate(zip(original_text,final_text),1):
        assert new==(old.replace('v','V') if i in (200,202,203,204,232) else old),(i,old,new)
    assert texts(preview(before,False))==texts(preview(doc,False))
    # Reject only the six new letter edits: the original full XML must be restored.
    restored=doc.cloneNode(True)
    new_ids={str(i) for i in editor.new_revision_ids}
    for tag in ('ins','del'):
        for rev in list(restored.getElementsByTagNameNS(W,tag)):
            if rev.getAttributeNS(W,'id') not in new_ids:continue
            parent=rev.parentNode
            if tag=='del':
                for child in list(rev.childNodes):parent.insertBefore(child,rev)
            parent.removeChild(rev)
    assert restored.toxml()==before.toxml()
    assert len(doc.getElementsByTagNameNS(M,'oMath'))==355
    assert not any(r.getElementsByTagNameNS(M,'oMath') for tag in ('ins','del') for r in doc.getElementsByTagNameNS(W,tag))
    settings=D.parseString(package['word/settings.xml'])
    if not settings.getElementsByTagNameNS(W,'trackRevisions'):
        settings.documentElement.appendChild(settings.createElementNS(W,'w:trackRevisions'))
    updates={'word/document.xml':doc.toxml(encoding='UTF-8'),'word/settings.xml':settings.toxml(encoding='UTF-8')}
    draft=BASE/'11.9二稿_仅六处字母纠错_校验稿.docx'
    with zipfile.ZipFile(draft,'w') as z:
        for info in infos:z.writestr(info,updates.get(info.filename,package[info.filename]))
    with zipfile.ZipFile(draft) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if name.endswith(('.xml','.rels')):D.parseString(z.read(name))
            if name not in updates:assert z.read(name)==package[name]
    assert hashlib.sha256(source.read_bytes()).hexdigest()==current_sha
    staged=source.with_name(source.name+'.tmp')
    shutil.copy2(draft,staged);os.replace(staged,source)
    report={'source':str(source),'original_backup':str(backup),'replaces_broader_edit':True,
            'scope':'仅六个小写v改为已定义的大写V；其余内容恢复原稿',
            'author':'周奕博','edits':edits,'original_xml_restored_by_rejecting_only_new_edits':True,
            'all_original_revisions_preserved':True,'other_package_parts_unchanged_except_tracking_setting':True,
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'timestamp':datetime.now(timezone.utc).isoformat()}
    (BASE/'最终仅字母纠错核验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (BASE/'本次修改说明.txt').write_text('此前11段修订已撤回，以本次源文件为准。\n从修改前原稿重新制作，仅将6个错误向量符号v统一为已定义的大写V，修订署名周奕博。\n其余正文、金额公式、定义、措辞、排版、图片和既有修订均恢复原稿。\n核验：拒绝本次6个字母的修订，可完整恢复修改前的正文XML。\n源文件：'+str(source)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
