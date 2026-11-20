"""Add native Word comments to the intact patent, preserving all original content."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import xml.dom.minidom as DOM
import xml.etree.ElementTree as ET
import zipfile


BASE = Path(__file__).resolve().parent
SOURCE = BASE.parent / '01_专利文档/11.3-一稿-一种基于审计过程数据的报销审计系统及方法.docx'
OUTPUT = BASE.parent / '01_专利文档/11.3-一稿-专利全文_逐处批注版_20261104.docx'
REPORT = BASE / '11.3新版专利_全文修改意见.txt'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
W14 = 'http://schemas.microsoft.com/office/word/2010/wordml'
W15 = 'http://schemas.microsoft.com/office/word/2012/wordml'
CID = 'http://schemas.microsoft.com/office/word/2016/wordml/cid'
CEX = 'http://schemas.microsoft.com/office/word/2018/wordml/cex'
M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
AUTHOR = '专利修订建议'
STAMP = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def plain(node):
    if node.nodeType == node.ELEMENT_NODE:
        if node.namespaceURI in (W, M) and node.localName in ('t', 'delText'):
            return ''.join(c.data for c in node.childNodes if c.nodeType in (c.TEXT_NODE, c.CDATA_SECTION_NODE))
        if node.namespaceURI == W and node.localName == 'tab':
            return '\t'
    return ''.join(plain(c) for c in node.childNodes)


def element(doc, prefix, local, namespace, attrs=None):
    node = doc.createElementNS(namespace, f'{prefix}:{local}')
    for key, value in (attrs or {}).items():
        node.setAttributeNS(namespace, f'{prefix}:{key}', str(value))
    return node


def canonical(node):
    data = node.toxml() if hasattr(node, 'toxml') else node
    return ET.canonicalize(data)


def main():
    source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    report = REPORT.read_text()
    starts = list(re.finditer(r'(?m)^(\d{2})．([^\n]+)\n', report))
    issues = {}
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else report.index('\n附：给修订人员')
        block = report[match.end():end].strip()
        lines = [line for line in block.splitlines() if not line.startswith(('位置：', '说明：', '依据：'))]
        issues[int(match.group(1))] = {'title': match.group(2), 'text': '\n'.join(lines)}
    assert len(issues) == 37
    entries = []

    def add(number, paragraph, anchor='', body=None, end_paragraph=None):
        info = issues[number]
        text = body if body is not None else info['text']
        entries.append({'issue': number, 'paragraph': paragraph, 'end_paragraph': end_paragraph or paragraph,
                        'anchor': anchor, 'text': f'【修改建议{number:02d}】{info["title"]}\n{text}'})

    for p in (5, 41, 58, 113, 238):
        add(1, p, '仅对错误分量非零')
    for p in (7, 61):
        add(2, p, '确定性金额程序提取')
    add(2, 107, '由确定性金额程序检查')
    for p in (11, 65, 156):
        add(3, p, '相等')
    for p in (10, 64):
        add(4, p, '在必要信息角色均由证据提供')
    add(4, 164, '第e项材料提供必要角色')
    add(5, 125, '上述三个核心新增模块')
    for p in (177, 216, 222, 227, 231, 243, 245, 246):
        add(6, p, body='本段及全文统一按名称称呼：第一模块改为“关系判断模块”；第二模块改为“证据贡献核验子模块”；第三模块改为“局部校正子模块”；前两个模块改为“关系判断模块和证据贡献核验子模块”。三个可训练部分的总称统一为“三个可训练神经模块”。')
    add(7, 129, 'F由三个新增模块')
    add(7, 144)
    add(8, 152, '本说明书中i表示')
    add(9, 156, '政策条件集合P保存')
    add(10, 153, '每项证据记为')
    add(11, 158, body='对应此处已有的两条批注，中文解释补为：h_e为第e份证据的编码表示；h_ij^(0)为审计对象i与政策条件j的初始联合表示，i为审计对象索引，j为政策条件索引，上标(0)为初始轮次。完整编码公式中的h_i^X和h_j^P分别表示申请信息编码与政策条件编码，在下方“其中”段一并定义。')
    add(11, 161, 'hiX表示，hjP表示')
    add(12, 169, 'zije为，gije为')
    for p in (13, 28, 68, 91):
        add(12, p, '逐元素逻辑函数', body='将“逐元素逻辑函数”改为“逐元素Sigmoid函数”。在关系判断公式首次出现处定义：σ(x)=1/(1+exp(-x))。')
    add(13, 171, '其中')
    add(13, 164, '开放文本条件的m', body='将“开放文本条件的m也取0”改为“开放文本条件对应的m_ije取0”，与本段前面的完整下标一致。r_ije为确定性比较的三维独热结果，π_ije为模型输出的三类关系概率。')
    add(14, 176)
    add(15, 177, '对每个必要信息角色t取')
    add(16, 179, 'T为转置操作')
    add(17, 182)
    add(18, 183)
    add(18, 197, body=issues[18]['text'].split('补充：', 1)[1].split('表2三行', 1)[0].strip() + '\n本段v_ref、v_rel和v_dep同步写为v_ij^ref、v_ij^rel和v_ij^dep。')
    for p, name in ((188, 'ref'), (191, 'rel'), (194, 'dep')):
        add(18, p, body=f'本表符号改为v_ij^{name}，与表前条件级错误向量的下标一致。')
    add(19, 198, '按req')
    add(20, 199)
    add(21, 204, 'tanh为')
    add(22, 207)
    for p in (31, 98, 208):
        add(23, p)
    add(24, 209, '若材料支持上限K')
    add(25, 214, 'Aok为，Areview为，Ano为，Aclaim为')
    add(26, 215)
    add(27, 108)
    add(27, 220, '得到有效关系标签')
    add(27, 222, '使用有效y和a', body='将“使用有效y和a训练前两个模块及其投影层”改为“使用有效关系标签y_ije和支持侧、冲突侧引用标签y_ije^{ref,q}，训练关系判断模块、证据贡献核验子模块及其投影层”；q∈{S,C}。申请金额仍使用a_i表示。')
    add(28, 223, '依据变化后的事实生成')
    add(29, 227, 'δij为，ρ为扰动上限，∇为，·2为，ε防止分母为零，hij为')
    add(30, 246, '引用阈值为0.60')
    add(31, 248)
    add(32, 250, end_paragraph=253)
    add(32, 217, '适配监督包括', body='将“适配监督包括经核验的材料读取问答和训练集既有整单审核答案，分别用于材料读取和整单结果生成”改为：“材料读取LoRA使用经核验的材料读取问答监督；条件级模块使用经核验的关系、引用及事实状态监督；既有整单审核答案用于标签构建与整单结果评测。”本次交付的材料读取LoRA与三个条件级神经模块按实际训练分别描述。')
    add(32, 233, '整单监督以最终判定', body='将“整单监督以最终判定、最高金额及问题说明为输出目标，只对答案部分计算损失”改为：“既有整单审核答案用于构建条件级监督并评测整单结果；材料读取LoRA以经核验的读取问答为训练数据，仅对助手答案部分计算损失。”')
    add(32, 246, '长输入训练采用', body='本段删除“五卡长输入训练、前馈网络及归一化按token分块、跨设备梯度检查点”等旧整单训练描述。改为：“工程采用材料读取LoRA与三个可训练神经模块：适配后的底座用于材料读取和独立编码，关系判断及证据贡献核验采用条件级标签训练，局部校正采用自然错误反馈训练。”后接本段实际交付阈值。')
    add(33, 253, body='本次实际训练补充为：“本次局部校正训练使用经核验的自然错误反馈样本，以恢复条件关系、引用和状态为目标，对局部校正参数进行优化。”受限内部扰动保留为前文的样本构造方法；当前交付结果的训练描述采用本次实际使用的数据。')
    add(34, 256)
    add(34, 263, body='本表本发明系统的“60.9%”改为“52.17%（12/23）”。原始Qwen与未微调GLM均为“26.09%（6/23）”。表格建议增加“样本数”“答对数”，三个对照对象均使用23例固定调参验证集。')
    add(34, 266, body='本段整段改为：“微调后的完整报销审计系统联合正确率为52.17%，较原始Qwen3-VL-8B-Instruct和未微调GLM-4.1V-9B-Thinking均提高26.09个百分点。”\n下一行写：“数据：固定调参验证集，共23个报销工作流。”保留两项原始模型的对照数据。')
    add(35, 234, '如图4所示', body='将“如图4所示”改为“如图2所示”。正文附图说明及文末图片为图1、图2，图片本体保留。')
    add(35, 253, '第9节', body='将“按第9节所述方法训练”改为“按前述训练数据与训练方法进行训练”。当前正文只有五个系统模块小节，没有第9节。')
    add(36, 117, '通过保持屏蔽', body='将“通过保持屏蔽整份证据及其派生操作数与关系结果”改为“通过屏蔽整份证据及其派生操作数与关系结果”。')
    add(36, 173, '全部候选材料全部候选材料', body='将“先在全部候选材料全部候选材料下计算”改为“先在全部候选材料下计算”。')

    # Each defensive wording receives its own short, local replacement comment.
    styles = list(re.finditer(r'段落([\d、]+)(?:原文)：([^\n]+)\n改为：([^\n]+)', issues[37]['text']))
    for match in styles:
        ps = [int(v) for v in match.group(1).split('、')]
        old, new = match.group(2), match.group(3)
        for p in ps:
            anchor = {
                116: '而不是在置信度汇总层', 154: '文字识别结果中的匹配',
                156: '政策优先级、例外', 164: '材料读取可能出错',
                173: '不用仍可能输出高概率', 177: '不将0自动认定为无效引用',
                179: '不能由文件贡献直接证明', 182: '模型未读出不能直接证明',
                197: '错误向量为零表示', 205: '连续不变不能视为',
                218: '不能把这些预测也宣称', 227: '不能保证一次扰动',
                230: '不要求表示校正恢复缺失事实', 244: '模型没有读到入住人',
            }[p]
            add(37, p, anchor, body=f'此处原文：“{old}”\n建议改为：“{new}”')
    assert len(styles) == 13, len(styles)
    assert set(e['issue'] for e in entries) == set(range(1, 38))

    with zipfile.ZipFile(SOURCE) as z:
        package = {name: z.read(name) for name in z.namelist()}
        zipinfo = z.infolist()
    document = DOM.parseString(package['word/document.xml'])
    original_document = DOM.parseString(package['word/document.xml'])
    body = document.getElementsByTagNameNS(W, 'body')[0]
    paragraphs = body.getElementsByTagNameNS(W, 'p')
    comments = DOM.parseString(package['word/comments.xml'])
    extended = DOM.parseString(package['word/commentsExtended.xml'])
    ids = DOM.parseString(package['word/commentsIds.xml'])
    extensible = DOM.parseString(package['word/commentsExtensible.xml'])
    old_comment_ids = {int(n.getAttributeNS(W, 'id')) for n in comments.getElementsByTagNameNS(W, 'comment')}
    next_id = max(old_comment_ids) + 1
    para_ids = {n.getAttributeNS(W14, 'paraId') for n in document.getElementsByTagNameNS(W, 'p')}
    para_ids.update(n.getAttributeNS(W14, 'paraId') for n in comments.getElementsByTagNameNS(W, 'p'))
    durable_ids = {n.getAttributeNS(CID, 'durableId') for n in ids.getElementsByTagNameNS(CID, 'commentId')}
    added_ids = set()
    manifest = []
    unique_serial = 0

    def new_para_id():
        nonlocal unique_serial
        while True:
            unique_serial += 1
            value = f'{0x51000000 + unique_serial:08X}'
            if value not in para_ids:
                para_ids.add(value)
                return value

    def chosen_children(p, anchor):
        children = [n for n in p.childNodes if n.nodeType == n.ELEMENT_NODE and plain(n)]
        assert children, 'anchor paragraph has no visible text'
        if not anchor:
            return children[0], children[-1], plain(p)
        content = ''.join(plain(n) for n in children)
        position = content.find(anchor)
        assert position >= 0, (anchor, content)
        end = position + len(anchor)
        offset = 0
        selected = []
        for n in children:
            next_offset = offset + len(plain(n))
            if next_offset > position and offset < end:
                selected.append(n)
            offset = next_offset
        return selected[0], selected[-1], ''.join(plain(n) for n in selected)

    # Longest ranges go first; existing paragraph elements, text and math remain intact.
    entries.sort(key=lambda e: (e['paragraph'], -(e['end_paragraph'] - e['paragraph']), e['issue']))
    for e in entries:
        p = paragraphs[e['paragraph'] - 1]
        last_p = paragraphs[e['end_paragraph'] - 1]
        first, last, selected_text = chosen_children(p, e['anchor'])
        if last_p is not p:
            _, last, _ = chosen_children(last_p, '')
        comment_id = str(next_id)
        next_id += 1
        added_ids.add(comment_id)
        start = element(document, 'w', 'commentRangeStart', W, {'id': comment_id})
        stop = element(document, 'w', 'commentRangeEnd', W, {'id': comment_id})
        reference_run = element(document, 'w', 'r', W)
        properties = element(document, 'w', 'rPr', W)
        properties.appendChild(element(document, 'w', 'rStyle', W, {'val': 'af1'}))
        reference_run.appendChild(properties)
        reference_run.appendChild(element(document, 'w', 'commentReference', W, {'id': comment_id}))
        p.insertBefore(start, first)
        last_p.insertBefore(stop, last.nextSibling)
        last_p.insertBefore(reference_run, stop.nextSibling)

        comment = element(comments, 'w', 'comment', W, {'id': comment_id, 'author': AUTHOR, 'initials': '修订', 'date': STAMP})
        last_para_id = None
        for line in e['text'].splitlines():
            if not line.strip():
                continue
            cp = element(comments, 'w', 'p', W)
            last_para_id = new_para_id()
            cp.setAttributeNS(W14, 'w14:paraId', last_para_id)
            cp.setAttributeNS(W14, 'w14:textId', '77777777')
            pp = element(comments, 'w', 'pPr', W)
            pp.appendChild(element(comments, 'w', 'spacing', W, {'after': '80'}))
            cp.appendChild(pp)
            cr = element(comments, 'w', 'r', W)
            rp = element(comments, 'w', 'rPr', W)
            rp.appendChild(element(comments, 'w', 'rFonts', W, {'ascii': 'Cambria', 'hAnsi': 'Cambria', 'eastAsia': '宋体'}))
            rp.appendChild(element(comments, 'w', 'sz', W, {'val': '21'}))
            if line.startswith('【修改建议'):
                rp.appendChild(element(comments, 'w', 'b', W))
            cr.appendChild(rp)
            ct = element(comments, 'w', 't', W)
            ct.setAttributeNS('http://www.w3.org/XML/1998/namespace', 'xml:space', 'preserve')
            ct.appendChild(comments.createTextNode(line))
            cr.appendChild(ct)
            cp.appendChild(cr)
            comment.appendChild(cp)
        comments.documentElement.appendChild(comment)
        extended.documentElement.appendChild(element(extended, 'w15', 'commentEx', W15, {'paraId': last_para_id, 'done': '0'}))
        durable = f'{0x61000000 + int(comment_id):08X}'
        assert durable not in durable_ids
        durable_ids.add(durable)
        ids.documentElement.appendChild(element(ids, 'w16cid', 'commentId', CID, {'paraId': last_para_id, 'durableId': durable}))
        extensible.documentElement.appendChild(element(extensible, 'w16cex', 'commentExtensible', CEX, {'durableId': durable, 'dateUtc': STAMP}))
        manifest.append({**e, 'comment_id': int(comment_id), 'selected_text': selected_text})

    settings = DOM.parseString(package['word/settings.xml'])
    # Make existing and new comments visible without enabling tracked text edits.
    revision_view = settings.getElementsByTagNameNS(W, 'revisionView')
    if revision_view:
        view = revision_view[0]
    else:
        view = element(settings, 'w', 'revisionView', W)
        reference = settings.getElementsByTagNameNS(W, 'defaultTabStop')[0]
        settings.documentElement.insertBefore(view, reference)
    for key in ('markup', 'comments', 'insDel', 'formatting'):
        view.setAttributeNS(W, 'w:' + key, '1')

    modified = {
        'word/document.xml': document.toxml(encoding='UTF-8'),
        'word/comments.xml': comments.toxml(encoding='UTF-8'),
        'word/commentsExtended.xml': extended.toxml(encoding='UTF-8'),
        'word/commentsIds.xml': ids.toxml(encoding='UTF-8'),
        'word/commentsExtensible.xml': extensible.toxml(encoding='UTF-8'),
        'word/settings.xml': settings.toxml(encoding='UTF-8'),
    }
    with zipfile.ZipFile(OUTPUT, 'w') as z:
        for info in zipinfo:
            z.writestr(info, modified.get(info.filename, package[info.filename]))

    # Artifact checks: reversible additions only, every comment has a balanced anchor,
    # all original comments and every original equation, table, image and run survive.
    stripped = DOM.parseString(document.toxml(encoding='UTF-8'))
    for local in ('commentRangeStart', 'commentRangeEnd', 'commentReference'):
        for node in list(stripped.getElementsByTagNameNS(W, local)):
            if node.getAttributeNS(W, 'id') in added_ids:
                target = node.parentNode if local == 'commentReference' else node
                target.parentNode.removeChild(target)
    assert canonical(stripped) == canonical(original_document), 'Original document content or formatting changed'
    before_comments = DOM.parseString(package['word/comments.xml'])
    old_nodes = before_comments.getElementsByTagNameNS(W, 'comment')
    for old in old_nodes:
        existing = next(n for n in comments.getElementsByTagNameNS(W, 'comment') if n.getAttributeNS(W, 'id') == old.getAttributeNS(W, 'id'))
        assert old.toxml() == existing.toxml()
    for tag in ('commentRangeStart', 'commentRangeEnd', 'commentReference'):
        actual = [n.getAttributeNS(W, 'id') for n in document.getElementsByTagNameNS(W, tag)]
        assert len(actual) == len(set(actual))
        assert set(actual) == {str(i) for i in old_comment_ids} | added_ids
    with zipfile.ZipFile(OUTPUT) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if name.endswith('.xml') or name.endswith('.rels'):
                DOM.parseString(z.read(name))
            if name not in modified:
                assert z.read(name) == package[name]
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_sha
    checks = {
        'source': str(SOURCE), 'output': str(OUTPUT), 'source_sha256': source_sha,
        'output_sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        'original_comments': len(old_comment_ids), 'added_comments': len(entries),
        'issue_categories_covered': sorted({e['issue'] for e in entries}),
        'all_original_content_and_formatting_unchanged': True,
        'original_comments_unchanged': True, 'original_file_unchanged': True,
        'native_math_objects': len(document.getElementsByTagNameNS(M, 'oMath')),
        'images_unchanged': [name for name in package if name.startswith('word/media/')],
        'all_comment_anchors_balanced': True,
        'modified_package_parts': sorted(modified),
    }
    (BASE / '批注版核验.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    (BASE / '批注定位清单.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
