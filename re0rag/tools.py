from __future__ import annotations

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

from db.embedding import get_embedding_model
from db.literature_graph import graph_status, search_graph
from db.manager import (
    fetch_linked_tables,
    fetch_parents,
    keyword_search,
    keyword_search_tables,
    search,
    search_tables,
)

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
    """关键词检索 tool：Qdrant BM25 sparse search -> parent 召回。"""
    top_k = top_k or config.KEYWORD_TOP_K
    terms = _tokenize_query(query)
    child_hits = keyword_search(query=query, top_k=top_k)
    parents, _parent_ids = fetch_parents(child_hits)
    table_hits = keyword_search_tables(query=query, top_k=config.TABLE_TOP_K)
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
    "tool.graph_search",
    attributes=lambda args, kwargs, result: {
        "top_k": kwargs.get("top_k") or config.GRAPH_TOP_K,
        "evidence.count": len((result or {}).get("evidence") or []),
    },
)
def graph_search_tool(query: str, top_k: int | None = None) -> dict:
    """Search bibliographic relationships and return explainable graph paths."""
    status = graph_status()
    if not status.get("enabled"):
        return {
            "tool": "graph_search",
            "query": query,
            "documents": [],
            "evidence": [],
            "parents": [],
            "sources": [],
            "summary": f"图谱检索已关闭：{status.get('reason') or 'Neo4j 不可用'}",
        }
    evidence = search_graph(query, top_k=top_k or config.GRAPH_TOP_K)
    status = graph_status()
    if not status.get("enabled"):
        return {
            "tool": "graph_search",
            "query": query,
            "documents": [],
            "evidence": [],
            "parents": [],
            "sources": [],
            "summary": f"图谱检索已降级关闭：{status.get('reason') or 'Neo4j 不可用'}",
        }
    sources = []
    for index, item in enumerate(evidence, start=1):
        metadata = item.get("metadata") or {}
        title = metadata.get("title") or metadata.get("source") or "未知论文"
        paths = item.get("graph_paths") or []
        suffix = f" | {paths[0]}" if paths else ""
        sources.append(f"[G{index}] {title}{suffix}")
    return {
        "tool": "graph_search",
        "query": query,
        "documents": [],
        "evidence": evidence,
        "parents": [],
        "sources": sources,
        "summary": f"图谱检索返回 {len(evidence)} 篇论文及其关系路径",
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
    if action == "graph_search":
        return graph_search_tool(query)
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
