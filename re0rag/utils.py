from __future__ import annotations

"""
RAG 链路工具函数：Prompt 拼装、来源格式化、结构化 JSON 解析。
Prompt 模板从根 config 读取。
"""

import json
import re
import sys
from pathlib import Path

# 把项目根加入 sys.path 以便 import 根 config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = config.SYSTEM_PROMPT
USER_PROMPT_TEMPLATE = config.USER_PROMPT_TEMPLATE


def build_prompt() -> ChatPromptTemplate:
    """
    构造 LLM 用的 ChatPromptTemplate（system + user）。
    user 模板中含 {context} 与 {question} 两个槽位。
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT_TEMPLATE),
        ]
    )


def format_context(documents: list[dict]) -> str:
    """
    把召回的 parent 章节拼接成 prompt 中的 {context} 文本。
    每个召回 parent 标注序号、所属论文标题/期刊/作者/摘要,正文用 parent 全文
    (parent 表格已回填其原文,故整张表会随父召回一起进 context)。

    Args:
        documents: tool 层返回的 parent 列表(已 fetch_parents 召回去重)
                   每条含 content / metadata / parent_index / hit_child_indices

    Returns:
        拼接后的上下文字符串
    """
    if not documents:
        return "（无相关参考片段）"

    blocks = []
    for i, doc in enumerate(documents, start=1):
        if doc.get("doc_type") == "table" or doc.get("metadata", {}).get("doc_type") == "table":
            blocks.append(_format_table_context(i, doc))
            continue
        m = doc.get("metadata", {})
        title = m.get("title") or m.get("source", "未知来源")
        journal = m.get("journal")
        authors = m.get("authors") or []
        abstract = m.get("abstract") or ""
        content = doc.get("content", "")
        parent_index = doc.get("parent_index", 0)
        hit_idx = doc.get("hit_child_indices") or []

        header = f"[章节{i}] 论文: {title}"
        if journal:
            header += f" | 期刊: {journal}"
        if authors:
            header += f" | 作者: {', '.join(authors[:6])}"
        header += f" (parent {parent_index}, 命中子片段 {len(hit_idx)} 处)"

        body_parts = []
        if abstract:
            body_parts.append(f"该论文摘要: {abstract}")
        body_parts.append(f"章节内容:\n{content}")
        blocks.append(f"{header}\n" + "\n\n".join(body_parts))
    return "\n\n".join(blocks)


def _format_table_context(i: int, doc: dict) -> str:
    """把 table evidence 格式化进 LLM context。"""
    m = doc.get("metadata", {})
    title = m.get("title") or m.get("source", "未知来源")
    caption = doc.get("caption") or f"Table {doc.get('table_index', '?')}"
    columns = doc.get("columns") or []
    warnings = doc.get("parse_warnings") or []
    table_body = doc.get("markdown") or doc.get("table_content") or doc.get("content", "")
    cells = doc.get("cells") or []
    source_method = doc.get("source_method") or ""
    confidence = doc.get("confidence")
    context_before = doc.get("context_before") or ""
    context_after = doc.get("context_after") or ""

    header = f"[表格{i}] 论文: {title} | 表格: {caption}"
    parts = []
    if source_method:
        quality = f"抽取方式: {source_method}"
        if confidence:
            quality += f" | 置信度: {confidence}"
        parts.append(quality)
    if columns:
        parts.append("列名: " + ", ".join(columns))
    if warnings:
        parts.append("解析提示: " + ", ".join(warnings))
    if context_before:
        parts.append("表格前后相关正文(前): " + context_before)
    if context_after:
        parts.append("表格前后相关正文(后): " + context_after)
    if cells:
        parts.append("结构化单元格(JSON):\n" + json.dumps(cells, ensure_ascii=False))
    if table_body:
        parts.append("表格原文/候选内容:\n" + table_body)
    return header + "\n" + "\n\n".join(parts)


def format_evidence(evidence: list[dict]) -> str:
    """
    把 tool 返回的统一 evidence 拼接成回答 / 检查节点可用的上下文。
    当前 evidence 主要是 parent 章节，因此复用 format_context。
    """
    return format_context(evidence)


def format_sources(parents: list[dict], n_hits: int = 0) -> list[str]:
    """
    生成输出节点要展示的来源描述列表(基于召回的 parent 章节)。

    Args:
        parents: 召回的 parent 列表
        n_hits:  命中的 child 子片段总数(用于展示检索粒度)

    Returns:
        ["[1] 论文标题 (parent 0, 命中 2 处子片段)", ...]
    """
    sources = []
    for i, doc in enumerate(parents, start=1):
        m = doc.get("metadata", {})
        if doc.get("doc_type") == "table" or m.get("doc_type") == "table":
            title = m.get("title") or m.get("source", "未知来源")
            caption = doc.get("caption") or f"table {doc.get('table_index', '?')}"
            sources.append(f"[{i}] {title} / {caption}")
            continue
        title = m.get("title") or m.get("source", "未知来源")
        parent_index = doc.get("parent_index", 0)
        hit_count = len(doc.get("hit_child_indices") or [])
        sources.append(f"[{i}] {title} (parent {parent_index}, 命中 {hit_count} 处子片段)")
    if n_hits and not sources:
        sources.append(f"(命中 {n_hits} 处子片段但未召回到对应父章节)")
    return sources


def parse_llm_json(text: str, fallback: dict) -> dict:
    """
    尽量从 LLM 输出中解析 JSON；失败时返回 fallback。

    route / judge prompt 都要求只输出 JSON，但实际模型偶尔会包 markdown
    代码块或附加解释。这里做一层容错，避免图执行中断。
    """
    if not text:
        return fallback

    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)

    candidates = [raw]
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidates.append(raw[start:end + 1])

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            return data if isinstance(data, dict) else fallback
        except Exception:
            continue
    return fallback


def format_route_history(route_history: list[dict] | None) -> str:
    """把已尝试 route 历史格式化给 route prompt，帮助避免重复无效检索。"""
    if not route_history:
        return "（无）"
    lines = []
    for i, item in enumerate(route_history, start=1):
        action = item.get("action", "")
        query = item.get("query", "")
        reason = item.get("reason", "")
        lines.append(f"{i}. action={action}; query={query}; reason={reason}")
    return "\n".join(lines)


def format_judge_feedback(judge_result: dict | None) -> str:
    """把上一轮 judge 结果格式化给 route prompt。"""
    if not judge_result:
        return "（无）"
    return (
        f"passed={judge_result.get('passed')}; "
        f"answers_question={judge_result.get('answers_question')}; "
        f"answer_supported={judge_result.get('answer_supported')}; "
        f"has_hallucination={judge_result.get('has_hallucination')}; "
        f"reason={judge_result.get('reason')}; "
        f"suggested_action={judge_result.get('suggested_action')}; "
        f"suggested_query={judge_result.get('suggested_query')}"
    )
