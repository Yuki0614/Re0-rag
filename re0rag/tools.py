"""
Agentic RAG 本地工具层。

问答阶段可调用的 tool 都集中在这里，保持 db/ 目录只负责离线构建、
向量库管理和底层数据读写。每个 tool 返回统一结构，供回答节点拼接证据。
"""

import re
import sys
from pathlib import Path

# 把项目根加入 sys.path 以便 import 根 config 与 db 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

from db.chunk import load_split_file
from db.embedding import get_embedding_model
from db.manager import fetch_linked_tables, fetch_parents, search, search_tables

from trace.telemetry import trace_span

from .utils import format_sources


@trace_span(
    "tool.vector_search",
    attributes=lambda args, kwargs, result: {
        "top_k": kwargs.get("top_k") or config.VECTOR_TOP_K,
        "child_hits.count": len((result or {}).get("documents") or []),
        "parents.count": len((result or {}).get("parents") or []),
        "tables.count": len((result or {}).get("tables") or []),
        "max_score": max(
            [h.get("score", 0) for h in ((result or {}).get("documents") or [])],
            default=0,
        ),
    },
)
def vector_search_tool(query: str, top_k: int | None = None) -> dict:
    """
    向量检索 tool：query -> embedding -> Qdrant child 相似度检索 -> parent 召回。

    Args:
        query: 检索 query
        top_k: child 命中数量，默认为 config.VECTOR_TOP_K

    Returns:
        统一 tool result，含 child hits / parent evidence / sources
    """
    top_k = top_k or config.VECTOR_TOP_K
    model = get_embedding_model()
    query_vector = model.embed_query(query)
    child_hits = search(query_vector=query_vector, top_k=top_k, doc_type="text")
    parents, _parent_ids = fetch_parents(child_hits)
    table_hits = search_tables(query_vector=query_vector, top_k=config.TABLE_TOP_K)
    linked_tables = fetch_linked_tables(parents)
    tables = _merge_tables(linked_tables, table_hits)
    evidence = parents + tables
    sources = format_sources(evidence, n_hits=len(child_hits))

    return {
        "tool": "vector_search",
        "query": query,
        "query_vector": query_vector,
        "documents": child_hits,
        "evidence": evidence,
        "parents": parents,
        "tables": tables,
        "sources": sources,
        "summary": f"向量检索命中 {len(child_hits)} 个子片段，召回 {len(parents)} 个父章节，附加 {len(tables)} 张表格",
    }


@trace_span(
    "tool.keyword_search",
    attributes=lambda args, kwargs, result: {
        "top_k": kwargs.get("top_k") or config.KEYWORD_TOP_K,
        "terms.count": len((result or {}).get("terms") or []),
        "child_hits.count": len((result or {}).get("documents") or []),
        "parents.count": len((result or {}).get("parents") or []),
        "tables.count": len((result or {}).get("tables") or []),
        "max_score": max(
            [h.get("score", 0) for h in ((result or {}).get("documents") or [])],
            default=0,
        ),
    },
)
def keyword_search_tool(query: str, top_k: int | None = None) -> dict:
    """
    关键词检索 tool：扫描已落盘 children chunks，按词项命中打分，再召回 parent。

    该 tool 不修改 db/ 目录，也不建立新索引；它直接复用导入阶段已经生成的
    db/chunks/{stem}/{stem}_children.json 与 _parents.json。
    """
    top_k = top_k or config.KEYWORD_TOP_K
    terms = _tokenize_query(query)
    child_hits = _keyword_child_search(query=query, terms=terms, top_k=top_k)
    parents, _parent_ids = fetch_parents(child_hits)
    table_hits = _keyword_table_search(query=query, terms=terms, top_k=config.TABLE_TOP_K)
    linked_tables = fetch_linked_tables(parents)
    tables = _merge_tables(linked_tables, table_hits)
    evidence = parents + tables
    sources = format_sources(evidence, n_hits=len(child_hits))

    return {
        "tool": "keyword_search",
        "query": query,
        "terms": terms,
        "documents": child_hits,
        "evidence": evidence,
        "parents": parents,
        "tables": tables,
        "sources": sources,
        "summary": f"关键词检索命中 {len(child_hits)} 个子片段，召回 {len(parents)} 个父章节，附加 {len(tables)} 张表格",
    }


@trace_span("tool.no_retrieval")
def no_retrieval_tool(query: str) -> dict:
    """
    不检索 tool：用于寒暄、系统操作类问题，明确返回空证据。
    """
    return {
        "tool": "no_retrieval",
        "query": query,
        "documents": [],
        "evidence": [],
        "parents": [],
        "sources": [],
        "summary": "未检索：该问题被路由为不需要查询本地论文库",
    }


@trace_span(
    "tool.run",
    attributes=lambda args, kwargs, result: {
        "action": args[0] if args else kwargs.get("action"),
        "query.length": len(args[1] if len(args) > 1 else kwargs.get("query", "")),
    },
)
def run_tool(action: str, query: str) -> dict:
    """
    根据 route 节点给出的 action 调用对应 tool。
    非法 action 保守回退到向量检索。
    """
    if action == "keyword_search":
        return keyword_search_tool(query)
    if action == "no_retrieval":
        return no_retrieval_tool(query)
    return vector_search_tool(query)


def _tokenize_query(query: str) -> list[str]:
    """
    简单关键词切分：保留英文/数字术语与中文连续片段，过滤过短英文词。
    """
    raw_terms = re.findall(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]+", query.lower())
    stop_words = {
        "the", "and", "for", "with", "what", "how", "why", "is", "are", "of",
        "to", "in", "on", "a", "an", "does", "do", "did", "论文", "什么",
        "怎么", "如何", "哪些", "这个", "那个",
    }
    terms = []
    for term in raw_terms:
        if term in stop_words:
            continue
        if re.fullmatch(r"[a-z]+", term) and len(term) <= 2:
            continue
        terms.append(term)
    return list(dict.fromkeys(terms))


def _keyword_child_search(query: str, terms: list[str], top_k: int) -> list[dict]:
    """
    在 children chunks 上做轻量关键词打分，返回与 db.manager.search 相同形状的 hits。
    """
    if not terms:
        terms = _tokenize_query(query)
    if not terms:
        return []

    query_lower = query.lower().strip()
    hits = []
    for child_file in Path(config.CHUNKS_DIR).glob("*/*_children.json"):
        try:
            children = load_split_file(child_file)
        except Exception:
            continue
        for child in children:
            content = child.get("content", "")
            metadata = child.get("metadata", {})
            haystack = _make_keyword_haystack(content, metadata)
            score = _score_keyword_hit(haystack, query_lower, terms)
            if score <= 0:
                continue
            hits.append(
                {
                    "content": content,
                    "metadata": metadata,
                    "child_index": child.get("child_index", 0),
                    "parent_id": child.get("parent_id"),
                    "parent_index": child.get("parent_index", 0),
                    "score": score,
                }
            )

    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:top_k]


def _keyword_table_search(query: str, terms: list[str], top_k: int) -> list[dict]:
    """扫描 tables JSON，按 caption/index_text/context 做关键词召回。"""
    if not terms:
        terms = _tokenize_query(query)
    if not terms:
        return []

    query_lower = query.lower().strip()
    hits = []
    for table_file in Path(config.CHUNKS_DIR).glob("*/*_tables.json"):
        try:
            tables = load_split_file(table_file)
        except Exception:
            continue
        for table in tables:
            haystack = _make_table_haystack(table)
            score = _score_keyword_hit(haystack, query_lower, terms)
            if score <= 0:
                continue
            rec = dict(table)
            rec["doc_type"] = "table"
            rec["score"] = score
            hits.append(rec)
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:top_k]


def _make_keyword_haystack(content: str, metadata: dict) -> str:
    """把正文与元数据合成关键词检索文本。"""
    parts = [
        content,
        metadata.get("source") or "",
        metadata.get("title") or "",
        metadata.get("journal") or "",
        " ".join(metadata.get("authors") or []),
    ]
    return "\n".join(parts).lower()


def _make_table_haystack(table: dict) -> str:
    metadata = table.get("metadata", {})
    parts = [
        table.get("caption") or "",
        table.get("index_text") or "",
        table.get("context_before") or "",
        table.get("context_after") or "",
        table.get("content") or "",
        metadata.get("source") or "",
        metadata.get("title") or "",
        metadata.get("journal") or "",
        " ".join(metadata.get("authors") or []),
    ]
    return "\n".join(parts).lower()


def _score_keyword_hit(haystack: str, query_lower: str, terms: list[str]) -> float:
    """轻量关键词评分：词项频次 + 完整 query 命中奖励。"""
    score = 0.0
    for term in terms:
        if re.fullmatch(r"[a-z0-9_\-]+", term):
            pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
            count = len(re.findall(pattern, haystack))
        else:
            count = haystack.count(term)
        if count:
            score += count * max(1.0, min(len(term), 12) / 4)
    if query_lower and query_lower in haystack:
        score += 8.0
    return score


def _merge_tables(*groups: list[dict]) -> list[dict]:
    merged = []
    seen = set()
    for group in groups:
        for table in group or []:
            metadata = table.get("metadata", {})
            source = metadata.get("source", "")
            key = (source, table.get("table_index"))
            if key in seen:
                continue
            seen.add(key)
            rec = dict(table)
            rec["doc_type"] = "table"
            merged.append(rec)
    return merged
