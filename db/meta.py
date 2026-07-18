from __future__ import annotations

"""
论文元数据抽取模块：在导入时用 LLM 从转好的 Markdown + PDF 首页文本抽取
标题 / 作者 / 期刊 / 摘要 四字段。

纯 LLM 路径：pymupdf4llm 已把 PDF 转成 Markdown，但部分论文（如 IEEE 双栏）
的 Abstract 段会被 pymupdf4llm 丢失，故额外用 fitz 直读 PDF 首页文本作为
补充源喂给 LLM。抽取完全交给 LLM 语义完成，不做任何版式规则判定。
跨格式（IEEE/arXiv/Elsevier/会议）鲁棒。
抽取参数与 prompt 模板从根 config 读取。
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional

# 把项目根加入 sys.path 以便 import 根 config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# 抽取参数（来自根 config）
META_DIR = config.META_DIR
HEAD_CHARS = config.META_LLM_HEAD_CHARS
FITZ_PAGES = getattr(config, "META_LLM_FITZ_PAGES", 2)


def extract_metadata(pdf_path: str | Path, md_path: str | Path) -> dict:
    """
    抽取单篇论文元数据（纯 LLM 从 Markdown 抽取）。

    Args:
        pdf_path: 源 PDF 路径（仅用于生成 source 名）
        md_path:  转好的 Markdown 路径（喂给 LLM）

    Returns:
        {source, pdf_stem, title, authors, journal, year, fields, keywords, abstract,
         method: {字段: llm|none}, confidence: {字段: ...}}
    """
    pdf_path = Path(pdf_path)
    md_path = Path(md_path)
    source = f"{pdf_path.stem}.md"

    meta = {
        "source": source,
        "pdf_stem": pdf_path.stem,
        "title": None, "authors": None,
        "journal": None, "year": None, "fields": None, "keywords": None,
        "abstract": None,
        "method": {k: "none" for k in ("title", "authors", "journal", "year", "fields", "keywords", "abstract")},
        "confidence": {k: 0.0 for k in ("title", "authors", "journal", "year", "fields", "keywords", "abstract")},
    }

    if not config.META_LLM_ENABLED:
        print("[meta] META_LLM_ENABLED=False，跳过抽取")
        return meta

    md_head = _read_head(md_path)
    fitz_text = _read_fitz_pages(pdf_path)
    evidence = _build_extraction_evidence(md_head, fitz_text)
    result = _llm_extract(evidence)
    if not result:
        return meta

    for k in ("title", "authors", "journal", "year", "fields", "keywords", "abstract"):
        val = result.get(k)
        if val:
            if isinstance(val, str):
                val = _clean_text(val)
                if k == "abstract":
                    val = _strip_abstract_prefix(val)
            meta[k] = val
            meta["method"][k] = "llm"
            meta["confidence"][k] = 0.85

    # 如果首轮没有摘要，做一次摘要专项重试。很多失败来自 PDF/Markdown 版式导致
    # Abstract 标记不标准，而标题/作者等字段已经抽到了。
    if not meta.get("abstract"):
        retry_abs = _llm_extract_abstract(evidence, meta)
        if retry_abs:
            meta["abstract"] = _strip_abstract_prefix(_clean_text(retry_abs))
            meta["method"]["abstract"] = "llm_retry"
            meta["confidence"]["abstract"] = 0.8

    # 反幻觉校验：摘要应能被原文宽松支持。这里做归一化和 token overlap，
    # 避免 Markdown 加粗碎片、PDF 断词、花引号等导致误杀。
    abs_text = meta.get("abstract") or ""
    if abs_text and not _abstract_supported(abs_text, evidence["validation_corpus"]):
        retry_abs = _llm_extract_abstract(evidence, meta)
        if retry_abs and _abstract_supported(retry_abs, evidence["validation_corpus"]):
            meta["abstract"] = _strip_abstract_prefix(_clean_text(retry_abs))
            meta["method"]["abstract"] = "llm_retry"
            meta["confidence"]["abstract"] = 0.8
        else:
            print("[meta] abstract 反幻觉校验未通过，置空")
            meta["abstract"] = None
            meta["method"]["abstract"] = "none"
            meta["confidence"]["abstract"] = 0.0

    return meta


def _read_head(md_path: Path) -> str:
    """读 Markdown 前 N 字符。"""
    try:
        return md_path.read_text(encoding="utf-8")[:HEAD_CHARS]
    except Exception:
        return ""


def _clean_text(s: str) -> str:
    """清洗 LLM 返回的字符串：去替换字符/私有区字符、折叠空白。"""
    # 去掉 Unicode 替换字符与 Private Use Area 等私有/不可打印字符
    s = "".join(ch for ch in s
                if not (0xE000 <= ord(ch) <= 0xF8FF or ord(ch) == 0xFFFD))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _strip_abstract_prefix(s: str) -> str:
    """去掉摘要被连带摘入的 'Abstract' 前缀（如 'Abstract—...' / 'Abstract: ...'）。"""
    m = re.match(
        r"^\s*(?:abstract|a\s+b\s+s\s+t\s+r\s+a\s+c\s+t|摘要)\b\s*[:：—\-–]*\s*",
        s,
        re.I,
    )
    if m:
        s = s[m.end():]
    return s.strip()


def _read_fitz_pages(pdf_path: Path, max_pages: int = FITZ_PAGES) -> str:
    """用 fitz 直读 PDF 前几页文本，作 pymupdf4llm 丢失段的补充源。"""
    try:
        import fitz  # PyMuPDF，按需导入；缺失时 Markdown 路径仍可测试/运行

        doc = fitz.open(str(pdf_path))
        pages = []
        for i in range(min(max_pages, len(doc))):
            pages.append(doc[i].get_text())
        doc.close()
        return "\n\n".join(pages)[:HEAD_CHARS]
    except ModuleNotFoundError:
        return _read_pypdf_pages(pdf_path, max_pages)
    except Exception as e:
        print(f"[meta] fitz 读 PDF 前几页失败: {e}")
        return _read_pypdf_pages(pdf_path, max_pages)


def _read_pypdf_pages(pdf_path: Path, max_pages: int = FITZ_PAGES) -> str:
    """fitz 不可用时，用 PyPDF2 兜底读取 PDF 前几页文本。"""
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages[:max_pages]:
            pages.append(page.extract_text() or "")
        text = "\n\n".join(pages)[:HEAD_CHARS]
        if text.strip():
            print("[meta] 未安装 PyMuPDF(fitz)，已用 PyPDF2 读取 PDF 前几页")
        return text
    except ModuleNotFoundError:
        print("[meta] 未安装 PyMuPDF(fitz)/PyPDF2，跳过 PDF 首页补充源")
        return ""
    except Exception as e:
        print(f"[meta] PyPDF2 读 PDF 前几页失败: {e}")
        return ""


def _build_extraction_evidence(md_head: str, fitz_text: str) -> dict:
    """把原始 Markdown/PDF 文本整理成 LLM 更容易使用的证据包。"""
    candidates = []
    candidates.extend(_find_abstract_windows(md_head, "markdown"))
    candidates.extend(_find_abstract_windows(fitz_text, "pdf"))

    if not candidates:
        front = _front_matter_window(md_head)
        if front:
            candidates.append("[markdown front-matter]\n" + front)

    abstract_candidates = "\n\n".join(candidates[:6]) or "（未定位到显式摘要候选）"
    return {
        "md_head": md_head,
        "fitz_text": fitz_text,
        "abstract_candidates": abstract_candidates,
        "validation_corpus": " ".join([md_head or "", fitz_text or "", abstract_candidates]),
    }


def _find_abstract_windows(text: str, label: str) -> list[str]:
    """从 Abstract / A B S T R A C T / 摘要 标记附近截取候选窗口。"""
    if not text:
        return []

    pattern = re.compile(r"(?i)(?:\babstract\b|a\s+b\s+s\s+t\s+r\s+a\s+c\s+t|摘要)")
    windows = []
    seen_spans = []
    for i, m in enumerate(pattern.finditer(text), start=1):
        start = max(0, m.start() - 500)
        end = min(len(text), m.end() + 4500)
        span = (start, end)
        if any(abs(start - s) < 200 for s, _ in seen_spans):
            continue
        seen_spans.append(span)
        snippet = _compact_for_prompt(text[start:end])
        windows.append(f"[{label} abstract-candidate {i}]\n{snippet}")
        if len(windows) >= 4:
            break
    return windows


def _front_matter_window(text: str) -> str:
    """没有显式 Abstract 时，给 LLM 一段题名/作者附近的首页前言窗口。"""
    if not text:
        return ""
    boundary = re.search(
        r"(?im)^\s*(?:#{1,3}\s*)?(?:\*+|_+)?\s*(?:1\.|1|i\.)\s*introduction\b",
        text,
    )
    end = boundary.start() if boundary else min(len(text), 5000)
    return _compact_for_prompt(text[:end])


def _compact_for_prompt(text: str, limit: int = 5000) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()[:limit]


def _llm_extract(evidence: dict) -> dict:
    """调用 LLM 从 Markdown + 首页文本抽四字段，双路降级。"""
    if not evidence.get("md_head") and not evidence.get("fitz_text"):
        return {}

    vars_dict = {
        "abstract_candidates": (evidence.get("abstract_candidates") or "（无）")[:8000],
        "md_head": (evidence.get("md_head") or "（无）")[:4000],
        "fitz_page0": (evidence.get("fitz_text") or "（无）")[:4000],
    }
    user_prompt = config.META_LLM_USER_PROMPT_TEMPLATE.format(**vars_dict)

    def _to_dict(r):
        if hasattr(r, "model_dump"):
            r = r.model_dump()
        if not isinstance(r, dict):
            return {}
        return {
            "title": r.get("title"), "authors": r.get("authors"),
            "journal": r.get("journal"), "year": r.get("year"),
            "fields": r.get("fields"), "keywords": r.get("keywords"),
            "abstract": r.get("abstract"),
        }

    try:
        from langchain_core.prompts import ChatPromptTemplate
        llm = _make_llm()
    except ModuleNotFoundError:
        print("[meta] LangChain 依赖缺失，降级为 OpenAI 兼容 HTTP 调用")
        return _to_dict(_post_chat_json(config.META_LLM_SYSTEM_PROMPT, user_prompt))

    prompt = ChatPromptTemplate.from_messages([
        ("system", config.META_LLM_SYSTEM_PROMPT),
        ("user", config.META_LLM_USER_PROMPT_TEMPLATE),
    ])

    # 优先结构化输出
    try:
        from pydantic import BaseModel, Field

        class PaperMeta(BaseModel):
            title: Optional[str] = Field(None)
            authors: Optional[list[str]] = Field(None)
            journal: Optional[str] = Field(None)
            year: Optional[int] = Field(None)
            fields: Optional[list[str]] = Field(None)
            keywords: Optional[list[str]] = Field(None)
            abstract: Optional[str] = Field(None)

        chain = prompt | llm.with_structured_output(PaperMeta)
        result = chain.invoke(vars_dict)
        return _to_dict(result)
    except Exception as e:
        print(f"[meta] 结构化抽取失败，降级 JSON 解析: {e}")

    # 降级：直接拿文本 json.loads
    try:
        chain = prompt | llm
        resp = chain.invoke(vars_dict)
        content = resp.content if hasattr(resp, "content") else str(resp)
        content = re.sub(r"^```(?:json)?|```$", "", content.strip(),
                         flags=re.M).strip()
        data = json.loads(content)
        return _to_dict(data)
    except Exception as e:
        print(f"[meta] LLM 抽取失败: {e}")
        return _to_dict(_post_chat_json(config.META_LLM_SYSTEM_PROMPT, user_prompt))


def _llm_extract_abstract(evidence: dict, meta: dict) -> str | None:
    """摘要为空或校验失败时，单独让 LLM 在候选证据里找摘要。"""
    system = (
        "你是论文摘要抽取助手。只从给定证据中摘录论文 abstract 原文。"
        "优先使用 Abstract / A B S T R A C T / 摘要 标记后的完整段落。"
        "不要总结、不要翻译、不要补写。找不到就返回 null。"
        "只输出严格 JSON：{\"abstract\": str|null}"
    )
    user = (
        "已识别标题：{title}\n"
        "摘要候选片段：\n{abstract_candidates}\n\n"
        "PDF 前几页文本：\n{fitz_page0}\n\n"
        "Markdown 前部文本：\n{md_head}\n\n"
        "请输出 JSON："
    )
    vars_dict = {
        "title": meta.get("title") or "（未知）",
        "abstract_candidates": (evidence.get("abstract_candidates") or "（无）")[:8000],
        "fitz_page0": (evidence.get("fitz_text") or "（无）")[:4000],
        "md_head": (evidence.get("md_head") or "（无）")[:4000],
    }
    user_prompt = user.format(**vars_dict)

    def _extract_value(data):
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        if isinstance(data, dict):
            value = data.get("abstract")
            return value if isinstance(value, str) and value.strip() else None
        return None

    try:
        from langchain_core.prompts import ChatPromptTemplate
        llm = _make_llm()
    except ModuleNotFoundError:
        print("[meta] LangChain 依赖缺失，摘要专项降级为 OpenAI 兼容 HTTP 调用")
        return _extract_value(_post_chat_json(system, user_prompt))

    prompt = ChatPromptTemplate.from_messages([("system", system), ("user", user)])

    try:
        from pydantic import BaseModel, Field

        class AbstractOnly(BaseModel):
            abstract: Optional[str] = Field(None)

        result = (prompt | llm.with_structured_output(AbstractOnly)).invoke(vars_dict)
        return _extract_value(result)
    except Exception as e:
        print(f"[meta] 摘要专项结构化抽取失败，降级 JSON 解析: {e}")

    try:
        resp = (prompt | llm).invoke(vars_dict)
        content = resp.content if hasattr(resp, "content") else str(resp)
        content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M).strip()
        return _extract_value(json.loads(content))
    except Exception as e:
        print(f"[meta] 摘要专项抽取失败: {e}")
        return _extract_value(_post_chat_json(system, user_prompt))


def _post_chat_json(system_prompt: str, user_prompt: str) -> dict:
    """不经 LangChain，直接调用 OpenAI 兼容 chat/completions 并解析 JSON。"""
    try:
        import requests
    except ModuleNotFoundError:
        print("[meta] requests 未安装，无法执行 HTTP LLM 降级")
        return {}

    try:
        url = config.LLM_BASE_URL.rstrip("/") + "/chat/completions"
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {config.LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": config.LLM_MODEL,
                "temperature": config.META_LLM_TEMPERATURE,
                "max_tokens": config.META_LLM_MAX_TOKENS,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=180,
        )
        resp.raise_for_status()
        choice = resp.json()["choices"][0]
        content = choice.get("message", {}).get("content") or ""
        content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M).strip()
        start, end = content.find("{"), content.rfind("}")
        if start >= 0 and end > start:
            content = content[start:end + 1]
        return json.loads(content) if content else {}
    except Exception as e:
        print(f"[meta] HTTP LLM 抽取失败: {e}")
        return {}


def _abstract_supported(abstract: str, corpus: str) -> bool:
    """宽松判断摘要是否能在原始证据中找到依据。"""
    if not abstract or not corpus:
        return False

    abs_norm = _normalize_for_match(abstract)
    corpus_norm = _normalize_for_match(corpus)
    probe = " ".join(abs_norm.split()[:18])
    if probe and probe in corpus_norm:
        return True

    abs_tokens = _content_tokens(abs_norm)[:80]
    if len(abs_tokens) < 10:
        return False
    corpus_tokens = set(_content_tokens(corpus_norm))
    overlap = sum(1 for t in abs_tokens if t in corpus_tokens) / len(abs_tokens)
    return overlap >= 0.68


def _normalize_for_match(text: str) -> str:
    text = re.sub(r"[*_`#>\[\]\(\)]+", " ", text)
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "", text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _content_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    stop = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
        "with",
    }
    return [t for t in tokens if len(t) > 2 and t not in stop]


def _make_llm():
    """轻量 LLM 工厂，复用 config 的 LLM 配置，抽取用低温。"""
    missing = [
        name for name, val in [
            ("LLM_BASE_URL", config.LLM_BASE_URL),
            ("LLM_API_KEY", config.LLM_API_KEY),
            ("LLM_MODEL", config.LLM_MODEL),
        ] if not val
    ]
    if missing:
        raise RuntimeError(
            "未配置 LLM，请在项目根目录 .env 或系统环境变量中设置: "
            + ", ".join(f"RE0RAG_{name}" for name in missing)
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        model=config.LLM_MODEL,
        temperature=config.META_LLM_TEMPERATURE,
        max_tokens=config.META_LLM_MAX_TOKENS,
    )


def save_metadata(meta: dict, output_path: str | Path) -> str:
    """把完整元数据写为 JSON（与 chunks.json 平级）。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(output_path)


def load_metadata(source_or_stem: str) -> dict | None:
    """按 source（=md 文件名）或 stem 加载已存的 meta.json。供 cmd_list 展示用。"""
    stem = Path(source_or_stem).stem
    meta_path = META_DIR / f"{stem}_meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
