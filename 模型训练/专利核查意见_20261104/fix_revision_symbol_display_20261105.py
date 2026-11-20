"""Repair Word revision visibility without accepting any content revisions.

Inline symbol explanations use ordinary Word runs with real sub/superscript
formatting. The two revised display equations keep native OMML and track their
actual changed math tokens inside the equation rather than enclosing oMath.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import xml.dom.minidom as D
import zipfile

from tracked_revision_20261105 import W, M, el, plain, preview

BASE = Path(__file__).resolve().parent
SOURCE = BASE.parent / '01_专利文档/11.3-一稿-一种基于审计过程数据的报销审计系统及方法.docx'
BACKUP_DIR = BASE / '字母显示修复前备份_20261105'
DRAFT = BASE / '周奕博_修订字母显示修复_20261105.docx'


def children(n, ns=None, name=None):
    return [c for c in n.childNodes if c.nodeType == c.ELEMENT_NODE
            and (ns is None or c.namespaceURI == ns)
            and (name is None or c.localName == name)]


def child(n, ns, name):
    return next(iter(children(n, ns, name)), None)


def scalar(n, ns, name, default):
    node = child(n, ns, name) if n is not None else None
    return node.getAttributeNS(ns, 'val') if node is not None else default


def math_chunks(n, level=0):
    """Return all displayed glyphs and script levels, including XML delimiters."""
    if n is None:
        return []
    if n.namespaceURI == W:
        if n.localName in ('t', 'delText'):
            return [(plain(n), level, False)]
        if n.localName in ('rPr', 'ctrlPr'):
            return []
    if n.namespaceURI != M:
        return [v for c in children(n) for v in math_chunks(c, level)]
    tag = n.localName
    if tag.endswith('Pr'):
        return []
    if tag == 't':
        return [(plain(n), level, False)]
    if tag == 'r':
        style = scalar(child(n, M, 'rPr'), M, 'sty', '')
        text = ''.join(plain(c) for c in children(n, M, 't'))
        return [(text, level, style != 'p')]
    if tag in ('sSub', 'sSup', 'sSubSup'):
        out = math_chunks(child(n, M, 'e'), level)
        if tag in ('sSub', 'sSubSup'):
            out += math_chunks(child(n, M, 'sub'), level - 1)
        if tag in ('sSup', 'sSubSup'):
            out += math_chunks(child(n, M, 'sup'), level + 1)
        return out
    if tag == 'd':
        props = child(n, M, 'dPr')
        beg = scalar(props, M, 'begChr', '(')
        end = scalar(props, M, 'endChr', ')')
        sep = scalar(props, M, 'sepChr', '|')
        out = [(beg, level, False)]
        for index, argument in enumerate(children(n, M, 'e')):
            if index:
                out.append((sep, level, False))
            out += math_chunks(argument, level)
        return out + [(end, level, False)]
    if tag == 'acc':
        out = math_chunks(child(n, M, 'e'), level)
        mark = scalar(child(n, M, 'accPr'), M, 'chr', '\u0302')
        for index, (text, script, italic) in enumerate(out):
            if text:
                out[index] = (text + mark, script, italic)
                break
        return out
    if tag == 'f':
        return [('(', level, False)] + math_chunks(child(n, M, 'num'), level) + [(')/(', level, False)] + math_chunks(child(n, M, 'den'), level) + [(')', level, False)]
    if tag == 'nary':
        sign = scalar(child(n, M, 'naryPr'), M, 'chr', '∫')
        return [(sign, level, False)] + math_chunks(child(n, M, 'sub'), level - 1) + math_chunks(child(n, M, 'sup'), level + 1) + math_chunks(child(n, M, 'e'), level)
    if tag == 'rad':
        return [('√(', level, False)] + math_chunks(child(n, M, 'e'), level) + [(')', level, False)]
    return [v for c in children(n) for v in math_chunks(c, level)]


def visible_text(n):
    if n.nodeType != n.ELEMENT_NODE:
        return ''
    if n.namespaceURI == M and n.localName in ('oMath', 'oMathPara'):
        return ''.join(text for text, _, _ in math_chunks(n))
    if n.namespaceURI == W and n.localName in ('t', 'delText'):
        return plain(n)
    if n.namespaceURI == W and n.localName in ('br', 'tab'):
        return '\n' if n.localName == 'br' else '\t'
    return ''.join(visible_text(c) for c in children(n))


def metadata_copy(doc, wrapper):
    copy = el(doc, 'w', wrapper.localName, W)
    for i in range(wrapper.attributes.length):
        a = wrapper.attributes.item(i)
        copy.setAttributeNS(a.namespaceURI, a.name, a.value)
    return copy


def text_run(doc, text, level, italic, deleted):
    run = el(doc, 'w', 'r', W)
    props = el(doc, 'w', 'rPr', W)
    props.appendChild(el(doc, 'w', 'rFonts', W, ascii='Cambria Math', hAnsi='Cambria Math', eastAsia='宋体'))
    if italic and any(c.isalpha() for c in text):
        props.appendChild(el(doc, 'w', 'i', W))
    if level:
        props.appendChild(el(doc, 'w', 'vertAlign', W, val='subscript' if level < 0 else 'superscript'))
        if abs(level) > 1:
            props.appendChild(el(doc, 'w', 'position', W, val=str((abs(level) - 1) * (-4 if level < 0 else 4))))
    run.appendChild(props)
    node = el(doc, 'w', 'delText' if deleted else 't', W)
    node.setAttributeNS('http://www.w3.org/XML/1998/namespace', 'xml:space', 'preserve')
    node.appendChild(doc.createTextNode(text))
    run.appendChild(node)
    return run


def core_pair(paragraph):
    wrappers = [c for c in children(paragraph, W) if c.localName in ('ins', 'del') and child(c, M, 'oMath') is not None]
    assert len(wrappers) == 2
    old = next(c for c in wrappers if c.localName == 'del')
    new = next(c for c in wrappers if c.localName == 'ins')
    assert len(children(old)) == len(children(new)) == 1
    return old, new


def fix_core_equations(doc, paragraphs):
    # Reference predictor: only the transpose symbol changed (T -> ⊤).
    p = paragraphs[177]
    old, new = core_pair(p)
    equation = child(old, M, 'oMath').cloneNode(True)
    target = next(t for t in equation.getElementsByTagNameNS(M, 't') if plain(t) == 'T')
    math_run = target.parentNode
    parent = math_run.parentNode
    deletion, insertion = metadata_copy(doc, old), metadata_copy(doc, new)
    deletion.appendChild(math_run.cloneNode(True))
    added_run = math_run.cloneNode(True)
    added_run.getElementsByTagNameNS(M, 't')[0].firstChild.data = '⊤'
    insertion.appendChild(added_run)
    parent.insertBefore(deletion, math_run)
    parent.insertBefore(insertion, math_run)
    parent.removeChild(math_run)
    p.insertBefore(equation, old)
    p.removeChild(old)
    p.removeChild(new)

    # Bounded perturbation: track only the changed radius multiplier.
    p = paragraphs[224]
    old, new = core_pair(p)
    equation = child(old, M, 'oMath').cloneNode(True)
    new_equation = child(new, M, 'oMath')
    target = next(t for t in equation.getElementsByTagNameNS(M, 't') if 'ρ' in plain(t))
    math_run = target.parentNode
    parent = math_run.parentNode
    before, after = plain(target).split('ρ', 1)
    if before:
        prefix = math_run.cloneNode(True)
        prefix.getElementsByTagNameNS(M, 't')[0].firstChild.data = before
        parent.insertBefore(prefix, math_run)
    deletion = metadata_copy(doc, old)
    removed = math_run.cloneNode(True)
    removed.getElementsByTagNameNS(M, 't')[0].firstChild.data = 'ρ'
    deletion.appendChild(removed)
    parent.insertBefore(deletion, math_run)
    insertion = metadata_copy(doc, new)
    nodes = children(new_equation)
    start = next(i for i, n in enumerate(nodes) if n.localName == 'sSub' and plain(child(n, M, 'e')) == 'ρ')
    stop = next(i for i, n in enumerate(nodes) if n.localName == 'f')
    radius_nodes = nodes[start:stop]
    # The trailing thin-space is unchanged, so keep it outside the insertion.
    if radius_nodes and plain(radius_nodes[-1]).isspace():
        radius_nodes = radius_nodes[:-1]
    assert ''.join(plain(n) for n in radius_nodes).startswith('ρrel')
    for node in radius_nodes:
        insertion.appendChild(node.cloneNode(True))
    parent.insertBefore(insertion, math_run)
    if after:
        suffix = math_run.cloneNode(True)
        suffix.getElementsByTagNameNS(M, 't')[0].firstChild.data = after
        parent.insertBefore(suffix, math_run)
    parent.removeChild(math_run)
    p.insertBefore(equation, old)
    p.removeChild(old)
    p.removeChild(new)


def paragraph_texts(doc):
    return [visible_text(p) for p in doc.getElementsByTagNameNS(W, 'body')[0].getElementsByTagNameNS(W, 'p')]


def main():
    sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    BACKUP_DIR.mkdir(exist_ok=True)
    backup = BACKUP_DIR / SOURCE.name
    if backup.exists():
        assert hashlib.sha256(backup.read_bytes()).hexdigest() == sha, 'Display-fix backup already exists for another version'
    else:
        shutil.copy2(SOURCE, backup)
    with zipfile.ZipFile(backup) as archive:
        info = archive.infolist()
        package = {name: archive.read(name) for name in archive.namelist()}
    doc = D.parseString(package['word/document.xml'])
    before = D.parseString(package['word/document.xml'])
    accepted_before = paragraph_texts(preview(before, True))
    rejected_before = paragraph_texts(preview(before, False))
    ps = doc.getElementsByTagNameNS(W, 'body')[0].getElementsByTagNameNS(W, 'p')
    fix_core_equations(doc, ps)
    converted = []
    for tag in ('ins', 'del'):
        for revision in list(doc.getElementsByTagNameNS(W, tag)):
            for equation in [n for n in children(revision, M) if n.localName in ('oMath', 'oMathPara')]:
                assert not equation.getElementsByTagNameNS(M, 'f'), 'A display fraction must stay native OMML'
                assert not equation.getElementsByTagNameNS(M, 'nary'), 'A display sum must stay native OMML'
                chunks = math_chunks(equation)
                assert chunks and ''.join(t for t, _, _ in chunks)
                for text, level, italic in chunks:
                    if text:
                        revision.insertBefore(text_run(doc, text, level, italic, tag == 'del'), equation)
                converted.append({'revision_id': revision.getAttributeNS(W, 'id'), 'kind': tag,
                                  'symbol_text': ''.join(t for t, _, _ in chunks)})
                revision.removeChild(equation)
    after_accepted = preview(doc, True)
    after_rejected = preview(doc, False)
    accepted_after = paragraph_texts(after_accepted)
    rejected_after = paragraph_texts(after_rejected)
    assert accepted_before == accepted_after, [(i+1, a, b) for i,(a,b) in enumerate(zip(accepted_before, accepted_after)) if a != b][:3]
    assert rejected_before == rejected_after, 'Rejecting revisions changed original visible symbols'
    # This exact unsupported nesting caused the symbols to disappear in Word.
    for tag in ('ins', 'del'):
        remaining = [r for r in doc.getElementsByTagNameNS(W, tag) if r.getElementsByTagNameNS(M, 'oMath')]
        assert not remaining, [(r.getAttributeNS(W, 'id'), [n.tagName for n in children(r)]) for r in remaining]
    all_revisions = list(doc.getElementsByTagNameNS(W, 'ins')) + list(doc.getElementsByTagNameNS(W, 'del'))
    assert all(n.getAttributeNS(W, 'author') == '周奕博' for n in all_revisions)
    assert len({n.getAttributeNS(W, 'id') for n in all_revisions}) == len(all_revisions)
    # Every symbol in the paragraph from the user's screenshot is actual text.
    path_paragraph = after_accepted.getElementsByTagNameNS(W, 'body')[0].getElementsByTagNameNS(W, 'p')[214]
    actual_text = ''.join(plain(t) for t in path_paragraph.getElementsByTagNameNS(W, 't'))
    assert all(value in actual_text for value in ('T=', 'NT', 'us=', 'ops', 'ins', 'outs', 'times'))
    protected_paragraphs = (128,159,160,167,168,170,174,175,178,199,201,202,203,210,211,212,213,225,226,229)
    accepted_ps = after_accepted.getElementsByTagNameNS(W, 'body')[0].getElementsByTagNameNS(W, 'p')
    for index in protected_paragraphs:
        assert accepted_ps[index-1].getElementsByTagNameNS(M, 'oMath'), index
    updated = doc.toxml(encoding='UTF-8')
    with zipfile.ZipFile(DRAFT, 'w') as archive:
        for item in info:
            archive.writestr(item, updated if item.filename == 'word/document.xml' else package[item.filename])
    with zipfile.ZipFile(DRAFT) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            if name.endswith('.xml') or name.endswith('.rels'):
                D.parseString(archive.read(name))
            if name != 'word/document.xml':
                assert archive.read(name) == package[name]
    record = {'source': str(SOURCE), 'backup': str(backup), 'source_sha256_before': sha,
              'draft': str(DRAFT), 'author': '周奕博', 'converted_inline_math_symbols': len(converted),
              'inserted_symbols': sum(c['kind']=='ins' for c in converted),
              'deleted_symbols': sum(c['kind']=='del' for c in converted),
              'accepted_visible_text_unchanged': True, 'rejected_visible_text_unchanged': True,
              'all_display_equations_still_native': True, 'no_equation_objects_enclosed_by_revision_wrappers': True,
              'path_paragraph_all_symbols_present_as_text': True,
              'original_comments_and_images_unchanged': True, 'written_to_source': False,
              'symbols': converted}
    (BASE/'字母显示修复核验_20261105.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    (BASE/'字母显示修复后_校对文本.txt').write_text('\n\n'.join(f'[{i+1:03}] {p}' for i,p in enumerate(accepted_after)))
    print(json.dumps({k:v for k,v in record.items() if k != 'symbols'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
