from __future__ import annotations

"""
文本切分模块：parent-child chunk + 表格独立抽取。

- 表格(markdown 管线表)整块抽出,单独存,不参与切分;在 parent 原文里回填表格内容,保证父召回时能看到整张表。
- 图片残留(![]() / <img>)直接丢弃(防止乱码)。loader 已 write_images=False,此处只做兜底清洗。
- parent = 按 PARENT_MAX_CHARS/PARENT_OVERLAP_CHARS 直接硬切全文。
- child = parent 内按 500/100 切;每个 child 带 parent_id 指回所属 parent。
- 检索用 child,召回用对应 parent(详见 db/manager.py、re0rag/nodes.py)。

切分参数统一从根 config 读取。
输出落到 db/chunks/{stem}/{stem}_tables.json / _parents.json / _children.json 三个聚合 JSON。
"""

import json
import re
import sys
import uuid
from pathlib import Path

# 把项目根加入 sys.path 以便 import 根 config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    class RecursiveCharacterTextSplitter:
        """轻量 fallback，便于缺依赖环境运行切分测试。"""

        def __init__(self, chunk_size, chunk_overlap=0, separators=None, length_function=len):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.length_function = length_function

        def create_documents(self, texts):
            class Doc:
                def __init__(self, page_content):
                    self.page_content = page_content

            docs = []
            step = max(1, self.chunk_size - self.chunk_overlap)
            for text in texts:
                start = 0
                while start < len(text):
                    docs.append(Doc(text[start:start + self.chunk_size]))
                    start += step
            return docs


# 切分参数（来自根 config）
CHUNK_SIZE = config.CHUNK_SIZE
CHUNK_OVERLAP = config.CHUNK_OVERLAP
PARENT_MAX_CHARS = config.PARENT_MAX_CHARS
PARENT_OVERLAP_CHARS = config.PARENT_OVERLAP_CHARS

# child 切分器（保留原中英混排分隔符优先级）
_CHILD_SEPARATORS = ["\n\n", "\n", "。", ".", " ", ""]

# 表格识别：一行 "|" 开头且其后紧跟一行形如 "|---|---|" 的分隔行
# 行首允许少量空白；表格块 = 连续的 "|" 开头行（含分隔行与数据行）。
_TABLE_ROW_RE = re.compile(r"^\s*\|")
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_TABLE_CAPTION_RE = re.compile(r"(?i)^\s*(?:table)\s+([ivxlcdm]+|\d+)\.?\s*(.*)$")
_FIGURE_RE = re.compile(r"(?i)^\s*(?:fig\.?|figure)\s+")
_SECTION_RE = re.compile(r"^\s*(?:#{1,6}\s+|[IVX]+\.\s+[A-Z]|[A-Z]\.\s+[A-Z])")

# 图片残留兜底清洗
_IMG_MD_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_IMG_HTML_RE = re.compile(r"<img[^>]*/?>")


def _stable_id(namespace: str, source: str, index: int) -> str:
    """由 source + 命名空间 + index 生成确定性 UUID v5。"""
    name = f"{source}::{namespace}::{index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


def _clean_images(text: str) -> str:
    """删除 markdown/HTML 图片标记，防止后续切分残留乱码。"""
    text = _IMG_MD_RE.sub("", text)
    text = _IMG_HTML_RE.sub("", text)
    return text


def _clean_inline_markup(s: str) -> str:
    """去掉常见 Markdown 强调标记，保留可读文本。"""
    s = re.sub(r"[*_`]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_caption_line(line: str) -> tuple[str, str] | None:
    """识别 Table 1 / TABLE V / **Table** **4** 等 caption 行。"""
    clean = _clean_inline_markup(line)
    m = _TABLE_CAPTION_RE.match(clean)
    if not m:
        return None
    table_no = m.group(1)
    rest = (m.group(2) or "").strip(" .:-")
    if rest.startswith(")") or rest.lower().startswith(
        ("summarizes", "shows", "reports", "presents", "illustrates", "is ", "are ")
    ):
        return None
    return table_no, rest


def _find_caption_near(lines: list[str], start: int) -> str:
    """在表格附近找 caption，优先表格前，再看表格后。"""
    for k in range(start, max(-1, start - 8), -1):
        parsed = _parse_caption_line(lines[k])
        if parsed:
            no, rest = parsed
            if rest:
                return f"Table {no}. {rest}"
            nxt = _next_nonempty_clean(lines, k + 1, limit=3)
            return f"Table {no}" + (f". {nxt}" if nxt else "")
    for k in range(start + 1, min(len(lines), start + 5)):
        parsed = _parse_caption_line(lines[k])
        if parsed:
            no, rest = parsed
            return f"Table {no}" + (f". {rest}" if rest else "")
    return ""


def _next_nonempty_clean(lines: list[str], start: int, limit: int = 3) -> str:
    for line in lines[start:start + limit]:
        clean = _clean_inline_markup(line)
        if clean:
            return clean
    return ""


def _parse_pipe_columns(header_line: str) -> list[str]:
    cells = [c.strip() for c in header_line.strip().strip("|").split("|")]
    return [_clean_inline_markup(c) for c in cells if _clean_inline_markup(c)]


def _extract_context(lines: list[str], start: int, end: int, window: int = 10) -> tuple[str, str]:
    before = "\n".join(lines[max(0, start - window):start]).strip()
    after = "\n".join(lines[end:min(len(lines), end + window)]).strip()
    return _clean_inline_markup(before), _clean_inline_markup(after)


def _table_placeholder(table_index: int, caption: str, columns: list[str]) -> str:
    label = caption or f"Table {table_index}"
    if columns:
        label += " | columns: " + ", ".join(columns[:8])
    return f"<TABLE_REF id={table_index} caption={json.dumps(label, ensure_ascii=False)} />"


def _make_table_record(
    source: str,
    table_index: int,
    content: str,
    caption: str,
    columns: list[str],
    context_before: str,
    context_after: str,
) -> dict:
    index_parts = [
        f"Source: {source}",
        f"Table: {caption or table_index}",
        "Columns: " + ", ".join(columns) if columns else "",
        "Context before: " + context_before if context_before else "",
        "Context after: " + context_after if context_after else "",
    ]
    index_text = "\n".join(p for p in index_parts if p)
    warnings = []
    if not content or len(content) < 80:
        warnings.append("short_or_caption_only")
    if columns and all(re.fullmatch(r"col\d+", c.lower()) for c in columns[: min(3, len(columns))]):
        warnings.append("generic_columns")
    return {
        "table_id": _stable_id("table", source, table_index),
        "content": content,
        "table_index": table_index,
        "caption": caption,
        "columns": columns,
        "context_before": context_before,
        "context_after": context_after,
        "index_text": index_text,
        "parse_warnings": warnings,
        "metadata": {"source": source, "doc_type": "table"},
    }


def _looks_like_plain_table_row(line: str) -> bool:
    clean = _clean_inline_markup(line)
    if not clean:
        return True
    if _parse_caption_line(clean) or _FIGURE_RE.match(clean) or _SECTION_RE.match(clean):
        return False
    if len(clean) > 180:
        return False
    has_metric = bool(re.search(r"\d|%|\bacc\b|\bf1\b|\bflops?\b|\bprecision\b|\brecall\b", clean, re.I))
    has_columns = len(re.split(r"\s{2,}|\t+", clean)) >= 2
    short_label = len(clean.split()) <= 10
    return has_metric or has_columns or short_label


def _caption_table_end(lines: list[str], start: int) -> int:
    """从 caption 行后尽量吞掉表格体，避免表格碎片进入 child。"""
    n = len(lines)
    end = start + 1

    # 兼容 TABLE V 下一行才是全大写 caption 的 IEEE 形式。
    while end < n and not _clean_inline_markup(lines[end]):
        end += 1
    if end < n:
        clean = _clean_inline_markup(lines[end])
        if clean and clean.upper() == clean and not _FIGURE_RE.match(clean):
            end += 1

    blank_run = 0
    collected_nonblank = 0
    while end < n and end < start + 45:
        clean = _clean_inline_markup(lines[end])
        if not clean:
            blank_run += 1
            end += 1
            if collected_nonblank >= 2 and blank_run >= 3:
                break
            continue
        blank_run = 0
        if _parse_caption_line(clean) or _FIGURE_RE.match(clean) or _SECTION_RE.match(clean):
            break
        if collected_nonblank >= 2 and not _looks_like_plain_table_row(clean):
            break
        if not _looks_like_plain_table_row(clean):
            break
        collected_nonblank += 1
        end += 1
    return max(start + 1, end)


def _extract_tables(text: str, source: str) -> tuple[list[dict], str]:
    """
    抽取表格证据,返回 (tables, text_without_table_body)。

    tables: [{table_id, content, caption, columns, context_before, context_after, index_text, ...}]
    text_without_table_body: 原文中的表格块替换为短 TABLE_REF，占位保留关联，不让表格正文污染 child embedding。
    """
    lines = text.split("\n")
    tables: list[dict] = []
    out_lines = []
    table_index = 0
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        # 命中 Markdown 管线表：当前行 | 开头，且下一行是分隔行 |---|
        if _TABLE_ROW_RE.match(line) and i + 1 < n and _TABLE_SEP_RE.match(lines[i + 1]):
            block = [line, lines[i + 1]]
            j = i + 2
            while j < n and _TABLE_ROW_RE.match(lines[j]):
                block.append(lines[j])
                j += 1
            content = "\n".join(block)
            caption = _find_caption_near(lines, i)
            columns = _parse_pipe_columns(line)
            context_before, context_after = _extract_context(lines, i, j)
            tables.append(_make_table_record(
                source, table_index, content, caption, columns, context_before, context_after
            ))
            out_lines.append(_table_placeholder(table_index, caption, columns))
            table_index += 1
            i = j
            continue

        parsed = _parse_caption_line(line)
        if parsed:
            table_no, rest = parsed
            j = _caption_table_end(lines, i)
            block = lines[i:j]
            caption = f"Table {table_no}" + (f". {rest}" if rest else "")
            if not rest:
                nxt = _next_nonempty_clean(lines, i + 1, limit=4)
                if nxt:
                    caption += f". {nxt}"
            content = "\n".join(block).strip()
            context_before, context_after = _extract_context(lines, i, j, window=12)
            tables.append(_make_table_record(
                source, table_index, content, caption, [], context_before, context_after
            ))
            out_lines.append(_table_placeholder(table_index, caption, []))
            table_index += 1
            i = j
            continue

        out_lines.append(line)
        i += 1
    return tables, "\n".join(out_lines)


def _split_parents(text: str, source: str) -> list[dict]:
    """
    按固定长度切 parent。

    这里不依赖标题结构：pymupdf4llm 对不同论文输出的标题层级不稳定，
    固定长度切分能让 parent 粒度更可控，也和当前项目的使用方式一致。
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_MAX_CHARS,
        chunk_overlap=PARENT_OVERLAP_CHARS,
        separators=["\n\n", "\n", "。", ".", " ", ""],
        length_function=len,
    )

    parents: list[dict] = []
    docs = parent_splitter.create_documents([text])
    for parent_index, doc in enumerate(d for d in docs if d.page_content.strip()):
        parents.append(
            {
                "parent_id": _stable_id("parent", source, parent_index),
                "content": doc.page_content.strip(),
                "parent_index": parent_index,
                "metadata": {"source": source},
            }
        )
    return parents


def _split_children(parents: list[dict], source: str) -> list[dict]:
    """在每个 parent 内按 500/100 切 child，每个 child 带 parent_id。"""
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=_CHILD_SEPARATORS,
        length_function=len,
    )
    children: list[dict] = []
    child_index = 0
    for p in parents:
        docs = child_splitter.create_documents([p["content"]])
        for d in docs:
            if not d.page_content.strip():
                continue
            children.append(
                {
                    "child_id": _stable_id("child", source, child_index),
                    "content": d.page_content,
                    "child_index": child_index,
                    "parent_id": p["parent_id"],
                    "parent_index": p["parent_index"],
                    "metadata": {"source": source},
                }
            )
            child_index += 1
    return children


def split_markdown(md_path: str | Path) -> dict:
    """
    将单个 Markdown 文件切分为  tables / parents / children 三类结构。

    Args:
        md_path: Markdown 文件路径

    Returns:
        {"tables": [...], "parents": [...], "children": [...]}
        - table:  {table_id, content, table_index, metadata:{source}}
        - parent: {parent_id, content, parent_index, metadata:{source, path}}
        - child:  {child_id, content, child_index, parent_id, parent_index, metadata:{source}}
    """
    md_path = Path(md_path)
    if not md_path.exists():
        raise FileNotFoundError(f"文件不存在: {md_path}")

    source = md_path.name
    text = md_path.read_text(encoding="utf-8")

    text = _clean_images(text)
    tables, text = _extract_tables(text, source)
    parents = _split_parents(text, source)

    # 给 parent 补 path 元信息
    base_meta = {"source": source, "path": str(md_path)}
    for p in parents:
        p["metadata"] = {**base_meta}

    children = _split_children(parents, source)
    return {"tables": tables, "parents": parents, "children": children}


def save_split(split_data: dict, paper_dir: str | Path, stem: str) -> dict:
    """
    把三类结构分别落盘为聚合 JSON。

    Args:
        split_data: split_markdown 返回结构
        paper_dir: 论文目录(db/chunks/{stem}/)
        stem:      论文 stem(文件名去后缀)，用作文件名前缀

    Returns:
        {"tables": path, "parents": path, "children": path}
    """
    paper_dir = Path(paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for key, suffix in [("tables", "_tables"), ("parents", "_parents"), ("children", "_children")]:
        out = paper_dir / f"{stem}{suffix}.json"
        out.write_text(
            json.dumps(split_data[key], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths[key] = str(out)
    return paths


def load_split_file(json_path: str | Path) -> list[dict]:
    """通用：读一个聚合 JSON（tables/parents/children 任一）。"""
    return json.loads(Path(json_path).read_text(encoding="utf-8"))
