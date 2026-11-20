"""Use standard paragraph-level Word revisions for the six symbol corrections."""
from pathlib import Path
from datetime import datetime, timezone
import sys, zipfile, json, hashlib
import xml.dom.minidom as D

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'专利核查意见_20261104'))
from simple_consistency_revision_20261106 import W,M,RevisionEditor,plain,preview,el
from fix_revision_symbol_display_20261105 import math_chunks,text_run,visible_text

SOURCE=BASE.parent/'01_专利文档/11.9-二稿-一种基于审计过程数据的报销审计系统及方法(1).docx'
BACKUP=BASE/'源文件修改前备份'/SOURCE.name
DRAFT=BASE/'仅六字母_标准Word修订及批注.docx'


def main():
    prior=json.loads((BASE/'公式修订显示修复核验.json').read_text())
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==prior['source_sha256']
    with zipfile.ZipFile(BACKUP) as z:
        infos=z.infolist();package={i.filename:z.read(i.filename) for i in infos}
    before=D.parseString(package['word/document.xml']);doc=D.parseString(package['word/document.xml'])
    e=RevisionEditor(doc)
    comments=D.Document();root=comments.createElementNS(W,'w:comments');root.setAttribute('xmlns:w',W);comments.appendChild(root)
    stamp=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    edits=[]
    for i,expected in ((200,1),(202,1),(203,1),(204,2),(232,1)):
        p=e.ps[i-1]
        equations=[m for m in p.getElementsByTagNameNS(M,'oMath') if any(s=='v' for s,_,_ in math_chunks(m))]
        assert len(equations)==expected
        for equation in equations:
            assert equation.parentNode is p
            chunks=math_chunks(equation)
            assert sum(s=='v' for s,_,_ in chunks)==1
            assert not any(equation.getElementsByTagNameNS(M,t) for t in ('f','nary','rad','acc'))
            for value,level,italic in chunks:
                if not value:continue
                if value!='v':
                    p.insertBefore(text_run(doc,value,level,italic,False),equation)
                    continue
                cid=len(edits)
                start=el(doc,'w','commentRangeStart',W,id=cid)
                end=el(doc,'w','commentRangeEnd',W,id=cid)
                p.insertBefore(start,equation)
                deletion=e.revision('del');deletion.appendChild(text_run(doc,'v',level,italic,True))
                insertion=e.revision('ins');insertion.appendChild(text_run(doc,'V',level,italic,False))
                p.insertBefore(deletion,equation);p.insertBefore(insertion,equation)
                p.insertBefore(end,equation)
                ref=el(doc,'w','r',W);props=el(doc,'w','rPr',W)
                props.appendChild(el(doc,'w','rStyle',W,val='CommentReference'));ref.appendChild(props)
                ref.appendChild(el(doc,'w','commentReference',W,id=cid));p.insertBefore(ref,equation)
                comment=el(comments,'w','comment',W,id=cid,author='周奕博',initials='ZYB',date=stamp)
                cp=el(comments,'w','p',W);cr=el(comments,'w','r',W);ct=el(comments,'w','t',W)
                ct.appendChild(comments.createTextNode('将错误向量的小写 v 改为大写 V，与前文定义统一。本处只改这一个字母。'))
                cr.appendChild(ct);cp.appendChild(cr);comment.appendChild(cp);root.appendChild(comment)
                edits.append({'paragraph':i,'comment_id':cid,'before':'v','after':'V'})
            p.removeChild(equation)
    assert len(edits)==6

    # Exact visible-content comparison in both accepted and rejected states.
    def texts(d):return [visible_text(p) for p in d.getElementsByTagNameNS(W,'body')[0].getElementsByTagNameNS(W,'p')]
    old=texts(preview(before,True));new=texts(preview(doc,True))
    for i,(a,b) in enumerate(zip(old,new),1):assert b==(a.replace('v','V') if i in (200,202,203,204,232) else a),(i,a,b)
    assert texts(preview(before,False))==texts(preview(doc,False))
    for tag in ('ins','del'):
        old_revs={n.getAttributeNS(W,'id'):n.toxml() for n in before.getElementsByTagNameNS(W,tag)}
        new_revs={n.getAttributeNS(W,'id'):n.toxml() for n in doc.getElementsByTagNameNS(W,tag)}
        assert all(new_revs[k]==v for k,v in old_revs.items())
        for rev in doc.getElementsByTagNameNS(W,tag):
            if rev.getAttributeNS(W,'author')=='周奕博':
                assert rev.parentNode.namespaceURI==W and rev.parentNode.localName=='p'
                assert len(rev.getElementsByTagNameNS(W,'r'))==1
                assert not rev.getElementsByTagNameNS(M,'oMath')
    assert len(doc.getElementsByTagNameNS(M,'oMath'))==349
    assert len(doc.getElementsByTagNameNS(W,'commentRangeStart'))==6
    assert len(doc.getElementsByTagNameNS(W,'commentRangeEnd'))==6
    assert len(doc.getElementsByTagNameNS(W,'commentReference'))==6

    settings=D.parseString(package['word/settings.xml'])
    if not settings.getElementsByTagNameNS(W,'trackRevisions'):
        settings.documentElement.appendChild(el(settings,'w','trackRevisions',W))
    view=settings.getElementsByTagNameNS(W,'revisionView')
    view=view[0] if view else el(settings,'w','revisionView',W)
    if not view.parentNode:settings.documentElement.appendChild(view)
    for key in ('markup','comments','insDel','formatting'):view.setAttributeNS(W,'w:'+key,'true')
    rels=D.parseString(package['word/_rels/document.xml.rels'])
    R='http://schemas.openxmlformats.org/package/2006/relationships'
    rel=rels.createElementNS(R,'Relationship');rel.setAttribute('Id','rIdZybSymbolComments')
    rel.setAttribute('Type','http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments');rel.setAttribute('Target','comments.xml')
    rels.documentElement.appendChild(rel)
    types=D.parseString(package['[Content_Types].xml'])
    typ=types.createElementNS('http://schemas.openxmlformats.org/package/2006/content-types','Override')
    typ.setAttribute('PartName','/word/comments.xml');typ.setAttribute('ContentType','application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml');types.documentElement.appendChild(typ)
    updates={'word/document.xml':doc.toxml(encoding='UTF-8'),'word/settings.xml':settings.toxml(encoding='UTF-8'),
             'word/_rels/document.xml.rels':rels.toxml(encoding='UTF-8'),'[Content_Types].xml':types.toxml(encoding='UTF-8'),
             'word/comments.xml':comments.toxml(encoding='UTF-8')}
    with zipfile.ZipFile(DRAFT,'w') as z:
        for info in infos:z.writestr(info,updates.get(info.filename,package[info.filename]))
        z.writestr('word/comments.xml',updates['word/comments.xml'])
    with zipfile.ZipFile(DRAFT) as z:
        assert z.testzip() is None
        for n in z.namelist():
            if n.endswith(('.xml','.rels')):D.parseString(z.read(n))
            if n not in updates:assert z.read(n)==package[n]
    report={'source':str(SOURCE),'draft':str(DRAFT),'sha_before':prior['source_sha256'],
            'six_letter_changes_only':True,'all_old_revisions_preserved':True,'edits':edits,
            'revision_author':'周奕博','new_word_text_revisions':12,'comments':6,
            'six_affected_equation_fragments_use_Word_runs_with_scripts':True,
            'accepted_and_rejected_visible_content_verified':True,'images_and_results_unchanged':True,
            'source_updated':False}
    (BASE/'普通Word修订与批注核验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
