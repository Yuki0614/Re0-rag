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
from html.parser import HTMLParser
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
_TABLE_REF_RE = re.compile(r"<TABLE_REF\s+id=(\d+)\b")
_HTML_TABLE_RE = re.compile(r"(?is)<table\b.*?</table>")
_FIGURE_RE = re.compile(r"(?i)^\s*(?:fig\.?|figure)\s+")
_SECTION_RE = re.compile(r"^\s*(?:#{1,6}\s+|[IVX]+\.\s+[A-Z]|[A-Z]\.\s+[A-Z])")
_MD_HEADING_RE = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$")
_ROMAN_SECTION_RE = re.compile(r"^\s*([IVXLCDM]+)\.\s+(.+?)\s*$")
_LETTER_SECTION_RE = re.compile(r"^\s*_?([A-Z])\._?\s+(.+?)\s*$")
_ITALIC_LETTER_SECTION_RE = re.compile(r"^\s*_([A-Z])\._\s+_(.+?)_\s*$")
_NUMBERED_SECTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+([A-Z][^\n]{2,})\s*$")
_ALL_CAPS_SECTION_RE = re.compile(r"^\s*([IVXLCDM]+)\.\s+([A-Z][A-Z0-9 ,:&()/\-]+)\s*$")

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
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
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
        (
            "summarizes", "shows", "reports", "presents", "illustrates",
            "displays", "lists", "contains", "compares", "is ", "are ",
            "specifically", "removing", "observed",
        )
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


def _parse_section_heading(line: str) -> tuple[int, str] | None:
    """Return (level, title) for common Markdown and paper section headings."""
    clean = _clean_inline_markup(line)
    if not clean:
        return None

    m = _MD_HEADING_RE.match(line)
    if m:
        title = _clean_inline_markup(m.group(2))
        if len(title) <= 2:
            return None
        return len(m.group(1)), title

    m = _ITALIC_LETTER_SECTION_RE.match(line)
    if m:
        return 2, f"{m.group(1)}. {_clean_inline_markup(m.group(2))}"

    m = _ALL_CAPS_SECTION_RE.match(clean)
    if m:
        return 1, f"{m.group(1)}. {_clean_inline_markup(m.group(2)).title()}"

    m = _ROMAN_SECTION_RE.match(clean)
    if m and clean.upper() == clean:
        return 1, f"{m.group(1)}. {_clean_inline_markup(m.group(2)).title()}"

    m = _LETTER_SECTION_RE.match(clean)
    if m:
        title = _clean_inline_markup(m.group(2))
        if "," in title or "“" in title or '"' in title or len(title.split()) > 10:
            return None
        return 2, f"{m.group(1)}. {title}"

    m = _NUMBERED_SECTION_RE.match(clean)
    if m:
        number = m.group(1)
        if "." not in number and len(number) > 2:
            return None
        level = min(6, 1 + number.count("."))
        return level, f"{number}. {_clean_inline_markup(m.group(2))}"

    return None


def _update_section_stack(
    stack: list[tuple[int, str]],
    level: int,
    title: str,
) -> list[tuple[int, str]]:
    stack = [(lvl, text) for lvl, text in stack if lvl < level]
    stack.append((level, title))
    return stack


def _split_text_by_limit(text: str, max_chars: int = PARENT_MAX_CHARS) -> list[str]:
    """Split an overlong section without crossing the configured parent size."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=PARENT_OVERLAP_CHARS,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return [doc.page_content.strip() for doc in splitter.create_documents([text]) if doc.page_content.strip()]


def _section_blocks(text: str) -> list[dict]:
    """Group text into heading-based blocks while preserving preface text."""
    lines = text.splitlines()
    blocks: list[dict] = []
    stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_meta = {
        "section": "Front Matter",
        "section_path": ["Front Matter"],
        "section_level": 0,
    }

    def flush() -> None:
        content = "\n".join(current_lines).strip()
        if content:
            blocks.append({"content": content, **current_meta})

    for line in lines:
        parsed = _parse_section_heading(line)
        if parsed:
            flush()
            level, title = parsed
            stack = _update_section_stack(stack, level, title)
            path = [text for _, text in stack]
            current_meta = {
                "section": title,
                "section_path": path,
                "section_level": level,
            }
            current_lines = [line]
        else:
            current_lines.append(line)
    flush()
    return blocks


def _parse_pipe_columns(header_line: str) -> list[str]:
    cells = [c.strip() for c in header_line.strip().strip("|").split("|")]
    return [_clean_inline_markup(c) for c in cells if _clean_inline_markup(c)]


def _parse_pipe_table(content: str) -> tuple[list[str], list[list[str]]]:
    lines = [line.strip() for line in content.splitlines() if line.strip().startswith("|")]
    if len(lines) < 2:
        return [], []
    raw_rows: list[list[str]] = []
    for line in lines:
        cells = [_clean_inline_markup(c.replace("<br>", " ")) for c in line.strip().strip("|").split("|")]
        raw_rows.append(cells)
    columns = [c for c in raw_rows[0]]
    rows = raw_rows[2:] if len(raw_rows) > 2 else []
    return columns, rows


class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._current_row = []
        elif tag in {"td", "th"} and self._current_row is not None:
            self._current_cell = []
            self._in_cell = True
        elif tag == "br" and self._in_cell and self._current_cell is not None:
            self._current_cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._current_row is not None and self._current_cell is not None:
            self._current_row.append(_clean_inline_markup("".join(self._current_cell)))
            self._current_cell = None
            self._in_cell = False
        elif tag == "tr" and self._current_row is not None:
            if any(cell.strip() for cell in self._current_row):
                self.rows.append(self._current_row)
            self._current_row = None

    def handle_data(self, data: str) -> None:
        if self._in_cell and self._current_cell is not None:
            self._current_cell.append(data)


def _parse_html_table(html: str) -> tuple[list[str], list[list[str]]]:
    parser = _HTMLTableParser()
    parser.feed(html)
    rows = parser.rows
    if not rows:
        return [], []
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    columns = padded[0]
    return columns, padded[1:]


def _cells_to_markdown(columns: list[str], rows: list[list[str]]) -> str:
    if not columns and not rows:
        return ""
    width = max([len(columns), *[len(row) for row in rows]], default=0)
    header = (columns or [f"Col{i + 1}" for i in range(width)]) + [""] * (width - len(columns))
    body = [row + [""] * (width - len(row)) for row in rows]

    def fmt(row: list[str]) -> str:
        return "|" + "|".join((cell or "").replace("\n", " ").strip() for cell in row[:width]) + "|"

    return "\n".join([fmt(header), "|" + "|".join("---" for _ in range(width)) + "|", *[fmt(row) for row in body]])


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
    *,
    source_method: str = "markdown",
    page_index: int | None = None,
    cells: list[list[str]] | None = None,
    confidence: float = 0.65,
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
    if not cells and not re.search(r"^\s*\|", content, re.M):
        warnings.append("unstructured_table_text")
    return {
        "table_id": _stable_id("table", source, table_index),
        "content": content,
        "markdown": content if re.search(r"^\s*\|", content, re.M) else "",
        "cells": cells or [],
        "table_index": table_index,
        "caption": caption,
        "columns": columns,
        "context_before": context_before,
        "context_after": context_after,
        "index_text": index_text,
        "parse_warnings": warnings,
        "source_method": source_method,
        "confidence": confidence,
        "metadata": {
            "source": source,
            "doc_type": "table",
            **({"page_index": page_index, "page": page_index + 1} if page_index is not None else {}),
        },
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


def _recent_table_ref(out_lines: list[str]) -> bool:
    for line in reversed(out_lines):
        if not line.strip():
            continue
        return bool(_TABLE_REF_RE.search(line))
    return False


def _next_table_ref(lines: list[str], start: int) -> bool:
    for line in lines[start:start + 8]:
        if not line.strip():
            continue
        return bool(_TABLE_REF_RE.search(line))
    return False


def _find_caption_before_text(text: str) -> str:
    lines = text.splitlines()
    start = max(0, len(lines) - 8)
    for line in reversed(lines[start:]):
        parsed = _parse_caption_line(line)
        if parsed:
            no, rest = parsed
            return f"Table {no}" + (f". {rest}" if rest else "")
    return ""


def _extract_html_tables(
    text: str,
    source: str,
    *,
    source_method: str,
    page_index: int | None,
    start_index: int,
) -> tuple[list[dict], str]:
    tables: list[dict] = []
    out_parts: list[str] = []
    cursor = 0
    table_index = start_index
    for match in _HTML_TABLE_RE.finditer(text):
        prefix = text[cursor:match.start()]
        html = match.group(0)
        columns, rows = _parse_html_table(html)
        markdown = _cells_to_markdown(columns, rows)
        caption = _find_caption_before_text(text[max(0, match.start() - 1200):match.start()])
        context_before, context_after = _extract_context(text.split("\n"), 0, 0)
        if markdown:
            tables.append(
                _make_table_record(
                    source,
                    table_index,
                    markdown,
                    caption,
                    columns,
                    context_before,
                    context_after,
                    source_method=source_method,
                    page_index=page_index,
                    cells=rows,
                    confidence=0.9 if source_method.startswith("ocr") else 0.8,
                )
            )
            out_parts.append(prefix)
            out_parts.append(_table_placeholder(table_index, caption, columns))
            table_index += 1
        else:
            out_parts.append(prefix)
            out_parts.append(html)
        cursor = match.end()
    if not tables:
        return [], text
    out_parts.append(text[cursor:])
    return tables, "".join(out_parts)


def _extract_tables(
    text: str,
    source: str,
    *,
    source_method: str = "markdown",
    page_index: int | None = None,
    start_index: int = 0,
) -> tuple[list[dict], str]:
    """
    抽取表格证据,返回 (tables, text_without_table_body)。

    tables: [{table_id, content, caption, columns, context_before, context_after, index_text, ...}]
    text_without_table_body: 原文中的表格块替换为短 TABLE_REF，占位保留关联，不让表格正文污染 child embedding。
    """
    html_tables, text = _extract_html_tables(
        text,
        source,
        source_method=source_method,
        page_index=page_index,
        start_index=start_index,
    )
    lines = text.split("\n")
    tables: list[dict] = list(html_tables)
    out_lines = []
    table_index = start_index + len(html_tables)
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
            columns, cells = _parse_pipe_table(content)
            context_before, context_after = _extract_context(lines, i, j)
            tables.append(_make_table_record(
                source,
                table_index,
                content,
                caption,
                columns,
                context_before,
                context_after,
                source_method=source_method,
                page_index=page_index,
                cells=cells,
                confidence=0.95 if source_method.startswith("ocr") else 0.85,
            ))
            out_lines.append(_table_placeholder(table_index, caption, columns))
            table_index += 1
            i = j
            continue

        parsed = _parse_caption_line(line)
        if parsed:
            if _recent_table_ref(out_lines) or _next_table_ref(lines, i + 1):
                out_lines.append(line)
                i += 1
                continue
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
                source,
                table_index,
                content,
                caption,
                [],
                context_before,
                context_after,
                source_method=source_method,
                page_index=page_index,
                confidence=0.45,
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


def _split_section_parents(text: str, source: str) -> list[dict]:
    """Build parent chunks from section blocks, splitting only overlong sections."""
    parents: list[dict] = []
    parent_index = 0
    for block in _section_blocks(text):
        pieces = _split_text_by_limit(block["content"], PARENT_MAX_CHARS)
        part_count = len(pieces)
        for section_chunk_index, piece in enumerate(pieces):
            content = piece.strip()
            if not content:
                continue
            metadata = {
                "source": source,
                "section": block.get("section", ""),
                "section_path": block.get("section_path", []),
                "section_level": block.get("section_level", 0),
                "section_chunk_index": section_chunk_index,
                "section_chunk_count": part_count,
            }
            if part_count > 1:
                metadata["section_part"] = f"{section_chunk_index + 1}/{part_count}"
            parents.append(
                {
                    "parent_id": _stable_id("parent", source, parent_index),
                    "content": content,
                    "parent_index": parent_index,
                    "metadata": metadata,
                }
            )
            parent_index += 1
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
                    "metadata": {
                        "source": source,
                        "section": p.get("metadata", {}).get("section", ""),
                        "section_path": p.get("metadata", {}).get("section_path", []),
                        "section_level": p.get("metadata", {}).get("section_level", 0),
                    },
                }
            )
            child_index += 1
    return children


def _normalize_text_key(text: str) -> str:
    text = _clean_inline_markup(text or "").lower()
    return re.sub(r"[^a-z0-9一-龥]+", "", text)


def _caption_key(caption: str) -> str:
    parsed = _parse_caption_line(caption or "")
    if parsed:
        return f"table-{parsed[0].lower()}"
    m = re.search(r"(?i)\btable\s+([ivxlcdm]+|\d+)\b", caption or "")
    return f"table-{m.group(1).lower()}" if m else ""


def _table_quality(table: dict) -> float:
    content = table.get("content") or ""
    cells = table.get("cells") or []
    warnings = set(table.get("parse_warnings") or [])
    score = float(table.get("confidence") or 0.0)
    if cells:
        score += min(1.0, len(cells) / 8)
    if re.search(r"^\s*\|", content, re.M):
        score += 0.8
    if len(content) > 300:
        score += 0.3
    if "short_or_caption_only" in warnings:
        score -= 1.0
    if "unstructured_table_text" in warnings:
        score -= 0.4
    return score


def _content_key(table: dict) -> str:
    cells = table.get("cells") or []
    if cells:
        preview = " ".join(" ".join(row) for row in cells[:4])
    else:
        preview = table.get("content") or ""
    return _normalize_text_key(preview)[:220]


def _finalize_table_refs(tables: list[dict], source: str) -> list[dict]:
    for table in tables:
        table_index = int(table.get("table_index") or 0)
        table["table_index"] = table_index
        table["table_id"] = _stable_id("table", source, table_index)
        table.setdefault("metadata", {})["source"] = source
        table["metadata"]["doc_type"] = "table"
    return tables


def _roman_to_int(value: str) -> int:
    value = value.lower()
    if value.isdigit():
        return int(value)
    nums = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    prev = 0
    for char in reversed(value):
        cur = nums.get(char, 0)
        total += -cur if cur < prev else cur
        prev = max(prev, cur)
    return total or 9999


def _table_sort_key(table: dict) -> tuple[int, int, int]:
    page = table.get("metadata", {}).get("page")
    page_order = int(page) if isinstance(page, int) else 9999
    m = re.search(r"(?i)\btable\s+([ivxlcdm]+|\d+)\b", table.get("caption") or "")
    table_no = _roman_to_int(m.group(1)) if m else 9999
    return page_order, table_no, int(table.get("table_index") or 0)


def _set_table_caption(table: dict, caption: str) -> None:
    table["caption"] = caption
    columns = table.get("columns") or []
    context_before = table.get("context_before") or ""
    context_after = table.get("context_after") or ""
    source = table.get("metadata", {}).get("source", "")
    index_parts = [
        f"Source: {source}",
        f"Table: {caption or table.get('table_index', 0)}",
        "Columns: " + ", ".join(columns) if columns else "",
        "Context before: " + context_before if context_before else "",
        "Context after: " + context_after if context_after else "",
    ]
    table["index_text"] = "\n".join(p for p in index_parts if p)


def _extract_ocr_tables(pdf_path: Path, source: str, start_index: int) -> tuple[list[dict], list[str]]:
    warnings: list[str] = []
    try:
        from db.ocr import extract_pdf_markdown_pages
    except Exception as exc:
        return [], [f"OCR 模块不可用，已跳过 OCR 表格增强: {exc}"]

    result = extract_pdf_markdown_pages(pdf_path)
    status = result.get("status")
    if status != "ok":
        warning = result.get("warning") or "OCR 未返回可用结果，已跳过 OCR 表格增强。"
        return [], [warning]

    tables: list[dict] = []
    next_index = start_index
    for page in result.get("pages") or []:
        page_markdown = page.get("markdown") or ""
        page_index = int(page.get("page_index") or 0)
        page_tables, _ = _extract_tables(
            _clean_images(page_markdown),
            source,
            source_method="ocr_paddle",
            page_index=page_index,
            start_index=next_index,
        )
        next_index += len(page_tables)
        tables.extend(page_tables)

    if not tables:
        warnings.append("OCR 已执行，但没有识别出可合并的表格。")
    elif result.get("from_cache"):
        warnings.append(f"OCR 使用缓存: {result.get('cache_path')}")
    else:
        warnings.append(f"OCR 表格增强完成，识别候选表格 {len(tables)} 张。")
    return tables, warnings


def _merge_table_candidates(markdown_tables: list[dict], ocr_tables: list[dict], source: str) -> list[dict]:
    merged: list[dict] = []
    by_caption: dict[str, int] = {}
    by_content: dict[str, int] = {}

    def add_or_replace(table: dict) -> None:
        cap_key = _caption_key(table.get("caption") or "")
        content_key = _content_key(table)
        keys = []
        if cap_key:
            keys.append(("caption", cap_key))
        if content_key:
            keys.append(("content", content_key))

        existing_index = None
        for key_type, key in keys:
            lookup = by_caption if key_type == "caption" else by_content
            if key in lookup:
                existing_index = lookup[key]
                break

        if existing_index is None:
            merged.append(table)
            idx = len(merged) - 1
        else:
            current = merged[existing_index]
            if _table_quality(table) > _table_quality(current):
                replacement = dict(table)
                replacement["table_index"] = current.get("table_index", table.get("table_index", 0))
                replacement["table_id"] = current.get("table_id") or _stable_id(
                    "table",
                    source,
                    int(replacement.get("table_index") or 0),
                )
                replacement.setdefault("metadata", {}).update(current.get("metadata", {}))
                replacement["metadata"]["source"] = source
                replacement["metadata"]["doc_type"] = "table"
                merged[existing_index] = replacement
            idx = existing_index

        if cap_key:
            by_caption[cap_key] = idx
        if content_key:
            by_content[content_key] = idx

    for table in markdown_tables:
        add_or_replace(table)
    for table in ocr_tables:
        # OCR caption-only fragments are rarely useful; keep only structured OCR tables.
        if table.get("cells") or re.search(r"^\s*\|", table.get("content") or "", re.M):
            add_or_replace(table)

    return _finalize_table_refs(sorted(merged, key=_table_sort_key), source)


def split_markdown(md_path: str | Path, pdf_path: str | Path | None = None) -> dict:
    """
    将单个 Markdown 文件切分为  tables / parents / children 三类结构。

    Args:
        md_path: Markdown 文件路径
        pdf_path: 原始 PDF 路径；提供后会尝试用 OCR 结果补强表格抽取

    Returns:
        {"tables": [...], "parents": [...], "children": [...], "warnings": [...]}
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
    warnings: list[str] = []

    if pdf_path is not None:
        ocr_tables, ocr_warnings = _extract_ocr_tables(Path(pdf_path), source, len(tables))
        warnings.extend(ocr_warnings)
        tables = _merge_table_candidates(tables, ocr_tables, source)

    parents = _split_section_parents(text, source)

    # 给 parent 补 path 元信息
    base_meta = {"source": source, "path": str(md_path)}
    for p in parents:
        p["metadata"] = {**p.get("metadata", {}), **base_meta}

    children = _split_children(parents, source)
    return {"tables": tables, "parents": parents, "children": children, "warnings": warnings}


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
