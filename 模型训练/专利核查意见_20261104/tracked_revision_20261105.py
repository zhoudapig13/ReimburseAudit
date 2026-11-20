"""Write reversible, author-attributed native Word revisions to the source patent."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import shutil
import xml.dom.minidom as D
import xml.etree.ElementTree as E
import zipfile

BASE = Path(__file__).resolve().parent
SOURCE = BASE.parent / '01_专利文档/11.3-一稿-一种基于审计过程数据的报销审计系统及方法.docx'
BACKUP = BASE / '源文件修订前备份_20261105' / SOURCE.name
DRAFT = BASE / '周奕博_原生修订校验稿_20261105.docx'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
AUTHOR = '周奕博'
DATE = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def plain(n):
    if n.nodeType == n.ELEMENT_NODE:
        if n.namespaceURI in (W, M) and n.localName in ('t', 'delText'):
            return ''.join(c.data for c in n.childNodes if c.nodeType == c.TEXT_NODE)
        if n.namespaceURI == W and n.localName in ('br', 'tab'):
            return '\n' if n.localName == 'br' else '\t'
    return ''.join(plain(c) for c in n.childNodes)


def el(doc, prefix, tag, ns, **attrs):
    n = doc.createElementNS(ns, f'{prefix}:{tag}')
    for k, v in attrs.items():
        n.setAttributeNS(ns, prefix + ':' + k, str(v))
    return n


@dataclass
class Unit:
    kind: str
    value: str
    node: object
    offset: int = 0

    @property
    def key(self):
        return (self.kind, self.value if self.kind == 'text' else self.node.toxml())


def units(nodes):
    result = []
    for n in nodes:
        if n.nodeType != n.ELEMENT_NODE or (n.namespaceURI == W and n.localName == 'pPr'):
            continue
        children = [c for c in n.childNodes if c.nodeType == c.ELEMENT_NODE]
        is_run = n.namespaceURI == W and n.localName == 'r'
        breaks = [c for c in children if c.namespaceURI == W and c.localName == 'lastRenderedPageBreak']
        if is_run and breaks and all(c.namespaceURI == W and c.localName in ('rPr', 't', 'delText', 'lastRenderedPageBreak') for c in children):
            for child in children:
                if child.localName == 'rPr':
                    continue
                fragment = n.cloneNode(True)
                for other in list(fragment.childNodes):
                    if other.nodeType == other.ELEMENT_NODE and other.localName != 'rPr':
                        fragment.removeChild(other)
                fragment.appendChild(child.cloneNode(True))
                value = plain(fragment)
                if child.localName == 'lastRenderedPageBreak':
                    result.append(Unit('fence', '', fragment))
                else:
                    result.extend(Unit('text', ch, fragment, i) for i, ch in enumerate(value))
            continue
        simple = is_run and all(c.namespaceURI == W and c.localName in ('rPr', 't', 'delText') for c in children)
        s = plain(n)
        if simple and s:
            result.extend(Unit('text', ch, n, i) for i, ch in enumerate(s))
        else:
            kind = 'math' if n.namespaceURI == M else ('opaque' if s else 'fence')
            result.append(Unit(kind, s, n))
    return result


class Editor:
    def __init__(self, document):
        self.doc = document
        self.ps = document.getElementsByTagNameNS(W, 'body')[0].getElementsByTagNameNS(W, 'p')
        self.before = [units(p.childNodes) for p in self.ps]
        self.after = [list(row) for row in self.before]
        self.cache = {}
        for p in self.ps:
            for node in p.getElementsByTagNameNS(M, 'oMath'):
                self.cache.setdefault(plain(node).replace(' ', ''), node)
        self.log = []
        self.covered = set()

    def text(self, p):
        return ''.join(u.value for u in self.after[p - 1])

    def run(self, text, p):
        n = el(self.doc, 'w', 'r', W)
        rp = None
        for old in self.before[p - 1]:
            if old.kind == 'text':
                candidates = old.node.getElementsByTagNameNS(W, 'rPr')
                rp = candidates[0].cloneNode(True) if candidates else None
                break
        if rp is None:
            rp = el(self.doc, 'w', 'rPr', W)
        for tag in ('highlight', 'color'):
            for child in list(rp.getElementsByTagNameNS(W, tag)):
                child.parentNode.removeChild(child)
        fonts = rp.getElementsByTagNameNS(W, 'rFonts')
        if fonts:
            fonts[0].setAttributeNS(W, 'w:eastAsia', '宋体')
        else:
            rp.insertBefore(el(self.doc, 'w', 'rFonts', W, eastAsia='宋体'), rp.firstChild)
        n.appendChild(rp)
        t = el(self.doc, 'w', 't', W)
        t.setAttributeNS('http://www.w3.org/XML/1998/namespace', 'xml:space', 'preserve')
        t.appendChild(self.doc.createTextNode(text))
        n.appendChild(t)
        return n

    def math(self, expression):
        """Simple OMML scripts; reuse existing native equations whenever possible."""
        pos = 0

        def mr(text):
            r = el(self.doc, 'm', 'r', M)
            t = el(self.doc, 'm', 't', M)
            t.appendChild(self.doc.createTextNode(text))
            r.appendChild(t)
            return r

        def parse(stop=False):
            nonlocal pos
            output = []
            while pos < len(expression):
                ch = expression[pos]
                if stop and ch == '}':
                    pos += 1
                    break
                if ch == '{':
                    pos += 1
                    group = parse(True)
                    delimiter = el(self.doc, 'm', 'd', M)
                    properties = el(self.doc, 'm', 'dPr', M)
                    properties.appendChild(el(self.doc, 'm', 'begChr', M, val='{'))
                    properties.appendChild(el(self.doc, 'm', 'endChr', M, val='}'))
                    delimiter.appendChild(properties)
                    content = el(self.doc, 'm', 'e', M)
                    for node in group:
                        content.appendChild(node)
                    delimiter.appendChild(content)
                    output.append(delimiter)
                    continue
                if ch == '\u0303' and output:
                    pos += 1
                    accent = el(self.doc, 'm', 'acc', M)
                    properties = el(self.doc, 'm', 'accPr', M)
                    properties.appendChild(el(self.doc, 'm', 'chr', M, val='\u0303'))
                    accent.appendChild(properties)
                    content = el(self.doc, 'm', 'e', M)
                    content.appendChild(output.pop())
                    accent.appendChild(content)
                    output.append(accent)
                    continue
                if ch in '_^':
                    mark = ch
                    pos += 1
                    if pos < len(expression) and expression[pos] == '{':
                        pos += 1
                        script = parse(True)
                    else:
                        script = [mr(expression[pos])]
                        pos += 1
                    assert output, expression
                    base = output.pop()
                    if mark == '^' and base.localName == 'sSub':
                        combined = el(self.doc, 'm', 'sSubSup', M)
                        for c in list(base.childNodes):
                            combined.appendChild(c.cloneNode(True))
                        sp = el(self.doc, 'm', 'sup', M)
                        for c in script:
                            sp.appendChild(c)
                        combined.appendChild(sp)
                    else:
                        combined = el(self.doc, 'm', 'sSub' if mark == '_' else 'sSup', M)
                        content = el(self.doc, 'm', 'e', M)
                        content.appendChild(base)
                        combined.appendChild(content)
                        sp = el(self.doc, 'm', 'sub' if mark == '_' else 'sup', M)
                        for c in script:
                            sp.appendChild(c)
                        combined.appendChild(sp)
                    output.append(combined)
                    continue
                # Keep multi-letter names in one run so _{...} binds the name.
                match = re.match(r'[A-Za-z]+|[^A-Za-z_^{\}]', expression[pos:])
                assert match, expression[pos:]
                token = match.group()
                pos += len(token)
                output.append(mr(token))
            return output

        root = el(self.doc, 'm', 'oMath', M)
        for n in parse():
            root.appendChild(n)
        cached = None if '\u0303' in expression else self.cache.get(plain(root).replace(' ', ''))
        return cached.cloneNode(True) if cached is not None else root

    def rich(self, text, p):
        ns = []
        for index, part in enumerate(text.split('$')):
            if not part:
                continue
            if index % 2:
                ns.append(self.math(part))
            else:
                lines = part.split('\n')
                for i, line in enumerate(lines):
                    if i:
                        r = el(self.doc, 'w', 'r', W)
                        r.appendChild(el(self.doc, 'w', 'br', W))
                        ns.append(r)
                    if line:
                        ns.append(self.run(line, p))
        assert len(text.split('$')) % 2 == 1
        return units(ns)

    def change(self, p, start, end, replacement, issue):
        row = self.after[p - 1]
        offset = 0
        selected = []
        for i, u in enumerate(row):
            next_offset = offset + len(u.value)
            if u.kind == 'fence':
                if start < offset < end:
                    selected.append(i)
            elif next_offset > start and offset < end:
                assert offset >= start and next_offset <= end, (p, u.value, start, end)
                selected.append(i)
            offset = next_offset
        if start == end:
            # Append before trailing bookmark/comment fences.
            idx = len(row)
            while idx and row[idx - 1].kind == 'fence':
                idx -= 1
            row[idx:idx] = replacement
        else:
            assert selected, (p, start, end)
            lo, hi = min(selected), max(selected) + 1
            old = row[lo:hi]
            # Keep all original comment/bookmark fences in the same order.
            # Proportional placement provides an anchor near the revised wording.
            fences = []
            current = 0
            old_length = sum(len(u.value) for u in old)
            for u in old:
                if u.kind == 'fence':
                    fences.append((round(current / max(old_length, 1) * len(replacement)), u))
                current += len(u.value)
            inserted = list(replacement)
            shift = 0
            for index, fence in fences:
                inserted.insert(index + shift, fence)
                shift += 1
            row[lo:hi] = inserted
        self.covered.add(issue)
        self.log.append({'paragraph': p, 'issue': issue, 'range': [start, end], 'inserted_text': ''.join(u.value for u in replacement)})

    def replace(self, p, old, new, issue, optional=False):
        text = self.text(p)
        position = text.find(old)
        if optional and position < 0:
            return False
        assert position >= 0, (p, old, text)
        self.change(p, position, position + len(old), self.rich(new, p), issue)
        return True

    def whole(self, p, text, issue):
        self.change(p, 0, len(self.text(p)), self.rich(text, p), issue)

    def append(self, p, text, issue):
        length = len(self.text(p))
        self.change(p, length, length, self.rich(text, p), issue)

    def transform_math(self, p, predicate, transform, issue):
        row = self.after[p - 1]
        count = 0
        for i, u in enumerate(row):
            if u.kind == 'math' and predicate(u.node):
                new = u.node.cloneNode(True)
                transform(new)
                row[i] = Unit('math', plain(new), new)
                count += 1
        assert count, p
        self.covered.add(issue)
        self.log.append({'paragraph': p, 'issue': issue, 'math_objects_changed': count})


def materialize(group, doc):
    out = []
    i = 0
    while i < len(group):
        u = group[i]
        if u.kind != 'text':
            out.append(u.node.cloneNode(True))
            i += 1
            continue
        j = i + 1
        while j < len(group) and group[j].kind == 'text' and group[j].node is u.node and group[j].offset == group[j - 1].offset + 1:
            j += 1
        text = ''.join(v.value for v in group[i:j])
        if text == plain(u.node):
            run = u.node.cloneNode(True)
        else:
            run = el(doc, 'w', 'r', W)
            for k in range(u.node.attributes.length):
                attr = u.node.attributes.item(k)
                run.setAttributeNS(attr.namespaceURI, attr.name, attr.value)
            rps = u.node.getElementsByTagNameNS(W, 'rPr')
            if rps:
                run.appendChild(rps[0].cloneNode(True))
            t = el(doc, 'w', 't', W)
            t.setAttributeNS('http://www.w3.org/XML/1998/namespace', 'xml:space', 'preserve')
            t.appendChild(doc.createTextNode(text))
            run.appendChild(t)
        out.append(run)
        i = j
    return out


def split_fences(row):
    chunks, fences = [[]], []
    for u in row:
        if u.kind == 'fence':
            fences.append(u)
            chunks.append([])
        else:
            chunks[-1].append(u)
    return chunks, fences


def revisions(editor):
    serial = 1000
    changes = {'ins': 0, 'del': 0}
    for index, p in enumerate(editor.ps):
        old, new = editor.before[index], editor.after[index]
        if [u.key for u in old] == [u.key for u in new]:
            continue
        a_chunks, a_fences = split_fences(old)
        b_chunks, b_fences = split_fences(new)
        assert [u.key for u in a_fences] == [u.key for u in b_fences]
        # pPr and all paragraph-level formatting stay exactly as in the source.
        for child in list(p.childNodes):
            if not (child.nodeType == child.ELEMENT_NODE and child.namespaceURI == W and child.localName == 'pPr'):
                p.removeChild(child)
        for chunk_index, (a, b) in enumerate(zip(a_chunks, b_chunks, strict=True)):
            diff = SequenceMatcher(None, [u.key for u in a], [u.key for u in b], autojunk=False)
            for op, i, j, k, l in diff.get_opcodes():
                if op == 'equal':
                    for n in materialize(a[i:j], editor.doc):
                        p.appendChild(n)
                    continue
                for tag, selected in (('del', a[i:j]), ('ins', b[k:l])):
                    if not selected:
                        continue
                    serial += 1
                    wrapper = el(editor.doc, 'w', tag, W, id=serial, author=AUTHOR, date=DATE)
                    for n in materialize(selected, editor.doc):
                        if tag == 'del':
                            for t in list(n.getElementsByTagNameNS(W, 't')):
                                editor.doc.renameNode(t, W, 'w:delText')
                        wrapper.appendChild(n)
                    p.appendChild(wrapper)
                    changes[tag] += 1
            if chunk_index < len(a_fences):
                p.appendChild(a_fences[chunk_index].node.cloneNode(True))
    return changes


def preview(document, accept):
    doc = D.parseString(document.toxml(encoding='utf-8'))
    for node in list(doc.getElementsByTagNameNS(W, 'ins')) + list(doc.getElementsByTagNameNS(W, 'del')):
        keep = (node.localName == 'ins') == accept
        parent = node.parentNode
        if keep:
            for child in list(node.childNodes):
                if not accept:
                    for t in list(child.getElementsByTagNameNS(W, 'delText')):
                        doc.renameNode(t, W, 'w:t')
                parent.insertBefore(child, node)
        parent.removeChild(node)
    return doc


def semantic(document):
    """Compare text with its formatting and intact nontext/math objects per paragraph."""
    body = document.getElementsByTagNameNS(W, 'body')[0]
    rows = []
    for p in body.getElementsByTagNameNS(W, 'p'):
        row = []
        for u in units(p.childNodes):
            if u.kind == 'text':
                rps = u.node.getElementsByTagNameNS(W, 'rPr')
                row.append(('text', u.value, rps[0].toxml() if rps else ''))
            else:
                row.append((u.kind, u.node.toxml()))
        pp = [c.toxml() for c in p.childNodes if c.nodeType == c.ELEMENT_NODE and c.namespaceURI == W and c.localName == 'pPr']
        rows.append((pp, row))
    return rows


def main():
    if not BACKUP.exists():
        BACKUP.parent.mkdir(parents=True)
        shutil.copy2(SOURCE, BACKUP)
    source_sha = hashlib.sha256(BACKUP.read_bytes()).hexdigest()
    assert source_sha == 'a51cba8525e0e5c475e095a47b4bc8667de866ca438c4d70550908a004ad8f6a'
    with zipfile.ZipFile(BACKUP) as archive:
        parts = {n: archive.read(n) for n in archive.namelist()}
        info = archive.infolist()
    document = D.parseString(parts['word/document.xml'])
    original = D.parseString(parts['word/document.xml'])
    edit = Editor(document)
    from tracked_revision_edits_20261105 import apply
    apply(edit)
    assert edit.covered == set(range(1, 38)), sorted(set(range(1, 38)) - edit.covered)
    counts = revisions(edit)
    rejected = preview(document, False)
    assert semantic(rejected) == semantic(original), 'Reject-all failed to restore original text/math/formatting'
    accepted = preview(document, True)
    accepted_ps = accepted.getElementsByTagNameNS(W, 'body')[0].getElementsByTagNameNS(W, 'p')
    for i, p in enumerate(accepted_ps):
        assert plain(p) == edit.text(i + 1), i + 1
    alltext = '\n'.join(plain(p) for p in accepted_ps)
    forbidden = ('更新错误分量为零的条件', '全部候选材料全部候选材料', '如图4所示', '第9节', '60.9%', '34.8个百分点', '1,438', 'hiX表示，hjP表示', 'tanh为，', 'q表示，S表示，C表示')
    assert not [s for s in forbidden if s in alltext]
    assert all(s in alltext for s in ('52.17%', '26.09%', '固定调参验证集', '116条', '0.65'))
    revs = list(document.getElementsByTagNameNS(W, 'ins')) + list(document.getElementsByTagNameNS(W, 'del'))
    assert all(n.getAttributeNS(W, 'author') == AUTHOR for n in revs)
    assert len({n.getAttributeNS(W, 'id') for n in revs}) == len(revs)
    settings = D.parseString(parts['word/settings.xml'])
    for tag in ('trackRevisions', 'revisionView'):
        for old in list(settings.getElementsByTagNameNS(W, tag)):
            old.parentNode.removeChild(old)
    reference = settings.getElementsByTagNameNS(W, 'defaultTabStop')[0]
    settings.documentElement.insertBefore(el(settings, 'w', 'revisionView', W, markup='1', comments='1', insDel='1', formatting='1'), reference)
    settings.documentElement.insertBefore(el(settings, 'w', 'trackRevisions', W, val='1'), reference)
    modified = {'word/document.xml': document.toxml(encoding='UTF-8'), 'word/settings.xml': settings.toxml(encoding='UTF-8')}
    with zipfile.ZipFile(DRAFT, 'w') as archive:
        for item in info:
            archive.writestr(item, modified.get(item.filename, parts[item.filename]))
    with zipfile.ZipFile(DRAFT) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            if name.endswith('.xml') or name.endswith('.rels'):
                D.parseString(archive.read(name))
            if name not in modified:
                assert archive.read(name) == parts[name]
    (BASE / '修订后阅读文本_校对用.txt').write_text('\n\n'.join(f'[{i+1:03}] {plain(p)}' for i, p in enumerate(accepted_ps)))
    (BASE / '源文件修订记录_20261105.json').write_text(json.dumps({'author': AUTHOR, 'source': str(SOURCE), 'backup': str(BACKUP), 'backup_sha256': source_sha, 'counts': counts, 'all_37_issue_categories_applied': True, 'reject_all_restores_original': True, 'all_other_package_parts_including_images_and_comments_unchanged': True, 'operations': edit.log}, ensure_ascii=False, indent=2))
    print(json.dumps({'draft': str(DRAFT), 'revisions': counts, 'paragraphs': len(accepted_ps), 'original_equations': len(original.getElementsByTagNameNS(M, 'oMath')), 'accepted_equations': len(accepted.getElementsByTagNameNS(M, 'oMath')), 'reject_all_restores_original': True, 'source_not_yet_replaced': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
