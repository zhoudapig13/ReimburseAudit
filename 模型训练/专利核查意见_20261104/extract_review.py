from pathlib import Path
import hashlib
import json
import zipfile
import xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parent
SOURCE = BASE.parent / '01_专利文档/11.3-一稿-一种基于审计过程数据的报销审计系统及方法.docx'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
NS = {'w': W, 'm': M}

def txt(el):
    if el is None:
        return ''
    tag = el.tag.split('}')[-1]
    if tag in ('del', 'moveFrom'):
        return ''
    if tag in ('t', 'delText'):
        return el.text or ''
    if tag in ('tab', 'br'):
        return ' '
    if tag.endswith('Pr'):
        return ''
    if tag in ('sSub', 'sSup', 'sSubSup'):
        out = txt(el.find('m:e', NS))
        for child, mark in [('sub', '_'), ('sup', '^')]:
            value = el.find('m:' + child, NS)
            if value is not None:
                out += mark + '{' + txt(value) + '}'
        return out
    if tag == 'f':
        return '(' + txt(el.find('m:num', NS)) + ')/(' + txt(el.find('m:den', NS)) + ')'
    if tag == 'd':
        pr = el.find('m:dPr', NS)
        def delim(name, default):
            child = pr.find('m:' + name, NS) if pr is not None else None
            return child.get('{' + M + '}val', default) if child is not None else default
        return delim('begChr', '(') + delim('sepChr', '|').join(txt(x) for x in el.findall('m:e', NS)) + delim('endChr', ')')
    if tag == 'acc':
        acc = el.find('m:accPr/m:chr', NS)
        mark = acc.get('{' + M + '}val', '̂') if acc is not None else '̂'
        return mark + '(' + txt(el.find('m:e', NS)) + ')'
    if tag == 'nary':
        op = el.find('m:naryPr/m:chr', NS)
        out = op.get('{' + M + '}val', '∫') if op is not None else '∫'
        for child, mark in [('sub', '_'), ('sup', '^')]:
            node = el.find('m:' + child, NS)
            if node is not None:
                out += mark + '{' + txt(node) + '}'
        return out + ' ' + txt(el.find('m:e', NS))
    if tag == 'rad':
        return 'sqrt(' + txt(el.find('m:e', NS)) + ')'
    return ''.join(txt(child) for child in el)

with zipfile.ZipFile(SOURCE) as archive:
    root = ET.fromstring(archive.read('word/document.xml'))
    body = root.find('w:body', NS)
    paragraphs = []
    for index, p in enumerate(body.findall('.//w:p', NS), 1):
        content = txt(p).strip()
        if content:
            comments = [x.get('{' + W + '}id') for x in p.findall('.//w:commentRangeStart', NS)]
            paragraphs.append({'paragraph': index, 'text': content, 'comments': comments})
    comments = []
    if 'word/comments.xml' in archive.namelist():
        for c in ET.fromstring(archive.read('word/comments.xml')):
            comments.append({'id': c.get('{' + W + '}id'), 'text': '\n'.join(txt(p) for p in c.findall('w:p', NS))})
    data = {'source': str(SOURCE), 'sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(), 'paragraphs': paragraphs, 'comments': comments}
    (BASE / '新版全文段落与公式.json').write_text(json.dumps(data, ensure_ascii=False, indent=2))
    (BASE / '新版全文段落与公式.txt').write_text('\n\n'.join(f"[{p['paragraph']:03}] {p['text']}" + (f" [批注{','.join(p['comments'])}]" if p['comments'] else '') for p in paragraphs) + '\n\n批注\n' + '\n\n'.join(f"[{c['id']}] {c['text']}" for c in comments))
print(json.dumps({'paragraphs': len(paragraphs), 'comments': len(comments), 'sha256': data['sha256']}, ensure_ascii=False))
