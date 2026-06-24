"""
RAG 链路节点：input / rewrite_query / route / tool / llm / judge / output。
每个节点接收 RAGState（或其子集），返回需要更新的状态字段。
检索能力通过 re0rag.tools 中的本地 tool 暴露给 route 节点。
LLM 与检索超参数统一从根 config 读取。
多轮对话：messages 累积历史，rewrite_node 消解指代后再检索。
"""

import sys
from pathlib import Path

# 把项目根加入 sys.path 以便 import 根 config 与 db 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .state import RAGState
from .tools import run_tool
from .utils import (
    build_prompt,
    format_evidence,
    format_judge_feedback,
    format_route_history,
    parse_llm_json,
)


def _check_llm_config() -> None:
    """配置校验：避免空值导致 OpenAI SDK 抛晦涩错误。"""
    missing = [
        name
        for name, val in [
            ("LLM_BASE_URL", config.LLM_BASE_URL),
            ("LLM_API_KEY", config.LLM_API_KEY),
            ("LLM_MODEL", config.LLM_MODEL),
        ]
        if not val
    ]
    if missing:
        raise RuntimeError(
            "未配置 LLM，请在项目根目录 .env 或系统环境变量中设置: "
            + ", ".join(f"RE0RAG_{name}" for name in missing)
        )


def _make_llm() -> ChatOpenAI:
    """用 OpenAI 兼容接口实例化生成模型。"""
    _check_llm_config()
    return ChatOpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
    )


def _format_history(messages: list) -> str:
    """
    把 messages 历史格式化为文本，供查询改写 prompt 使用。
    只取 HumanMessage / AIMessage 的文本内容，拼成“问/答”形式。
    """
    if not messages:
        return "（无）"
    lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"用户：{msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"助手：{msg.content}")
    return "\n".join(lines) if lines else "（无）"


def _history_before_current_question(messages: list, question: str) -> list:
    """
    取当前问题之前的对话历史。
    agentic 重试会在同一次 invoke 内产生候选答案，生成新答案时不把这些
    未通过检查的候选答案放回上下文，避免自我污染。
    """
    last_current_idx = None
    for i, msg in enumerate(messages or []):
        if isinstance(msg, HumanMessage) and msg.content.strip() == question.strip():
            last_current_idx = i
    if last_current_idx is None:
        return messages or []
    return (messages or [])[:last_current_idx]


def _as_bool(value, default: bool = False) -> bool:
    """把 LLM JSON 中可能出现的字符串布尔值归一化。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1", "是", "通过")
    return default


def input_node(state: RAGState) -> dict:
    """
    节点1：输入节点。
    确认 question 已存在（由调用方在 invoke 时传入），
    并把当前问题作为 HumanMessage 追加进 messages（多轮历史）。
    """
    question = state["question"].strip()
    if not question:
        raise ValueError("输入问题为空")
    print(f"\n[输入] 问题: {question}")
    return {
        "question": question,
        "retry_count": 0,
        "max_retries": config.AGENT_MAX_RETRIES,
        "route_history": [],
        "judge_result": {},
        "messages": [HumanMessage(content=question)],
    }


def rewrite_node(state: RAGState) -> dict:
    """
    节点2：查询改写节点。
    用对话历史 + 当前问题，让 LLM 生成一个独立、自包含的检索 query，
    消解指代（如“它”“这个”），使检索能查到正确内容。
    """
    question = state["question"]
    # messages 由 add_messages 累积，含本轮 HumanMessage 与历史
    history = _format_history(state.get("messages", [])[:-1])  # 去掉刚追加的本轮问题

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", config.REWRITE_SYSTEM_PROMPT),
            ("user", config.REWRITE_USER_PROMPT_TEMPLATE),
        ]
    )
    llm = _make_llm()
    chain = prompt | llm
    response = chain.invoke({"history": history, "question": question})
    rewritten = response.content.strip()

    print(f"[改写] 检索 query: {rewritten}")
    return {"rewritten_query": rewritten}


def route_node(state: RAGState) -> dict:
    """
    节点3：tool 路由节点。
    用 LLM 输出严格 JSON，选择 vector_search / keyword_search / no_retrieval。
    """
    question = state["question"]
    rewritten_query = state.get("rewritten_query") or question
    judge_feedback = format_judge_feedback(state.get("judge_result"))
    route_history_text = format_route_history(state.get("route_history"))

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", config.ROUTE_SYSTEM_PROMPT),
            ("user", config.ROUTE_USER_PROMPT_TEMPLATE),
        ]
    )
    llm = _make_llm()
    response = (prompt | llm).invoke(
        {
            "question": question,
            "rewritten_query": rewritten_query,
            "judge_feedback": judge_feedback,
            "route_history": route_history_text,
        }
    )

    fallback = {
        "action": "vector_search",
        "query": rewritten_query,
        "reason": "route JSON 解析失败，保守回退到向量检索",
    }
    decision = parse_llm_json(response.content, fallback=fallback)
    action = decision.get("action") or "vector_search"
    if action not in config.AGENT_ALLOWED_TOOLS:
        action = "vector_search"
    query = (decision.get("query") or rewritten_query or question).strip()
    reason = decision.get("reason") or ""

    decision = {"action": action, "query": query, "reason": reason}
    route_history = list(state.get("route_history") or [])
    route_history.append(decision)

    print(f"[路由] tool={action} query={query}")
    if reason:
        print(f"[路由] 原因: {reason}")

    return {
        "route_decision": decision,
        "selected_tool": action,
        "tool_query": query,
        "route_history": route_history,
    }


def tool_node(state: RAGState) -> dict:
    """
    节点4：执行 route_node 选择的本地 tool，并把结果归一化为 evidence。
    """
    action = state.get("selected_tool") or "vector_search"
    query = state.get("tool_query") or state.get("rewritten_query") or state["question"]
    result = run_tool(action, query)
    evidence = result.get("evidence", [])
    documents = result.get("documents", [])
    sources = result.get("sources", [])

    print(f"[工具] {result.get('summary')}")
    for s in sources:
        print(f"  - {s}")

    return {
        "tool_results": [result],
        "documents": documents,
        "parents": result.get("parents", []),
        "evidence": evidence,
        "sources": sources,
        "question_vector": result.get("query_vector", []),
    }


def llm_node(state: RAGState) -> dict:
    """
    节点4：LLM 节点。
    用 messages（含对话历史）+ 当前检索片段拼成 prompt，调用生成模型得到回答。
    回答追加为 AIMessage 进 messages，同时写入 answer 字段供输出节点取用。
    """
    question = state["question"]
    evidence = state.get("evidence", [])

    # 用 tool 返回的统一 evidence 拼 context(parent 内已回填表格原文)
    context = format_evidence(evidence)
    prompt = build_prompt()

    llm = _make_llm()

    # 用 messages（含历史）作为对话上下文，system 指令 + 检索片段作为本轮增强
    messages = prompt.format_messages(context=context, question=question)
    # 前置历史对话（不含本轮问题及本轮内部失败候选答案）
    history = _history_before_current_question(state.get("messages", []), question)
    full_messages = history + messages

    response = llm.invoke(full_messages)
    answer = response.content

    print(f"[生成] 回答长度: {len(answer)} 字符")
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


def judge_node(state: RAGState) -> dict:
    """
    节点6：答案检查节点。
    根据问题、证据和候选答案判断是否通过；失败时给 route 节点提供下一轮建议。
    """
    question = state["question"]
    answer = state.get("answer", "")
    context = format_evidence(state.get("evidence", []))

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", config.JUDGE_SYSTEM_PROMPT),
            ("user", config.JUDGE_USER_PROMPT_TEMPLATE),
        ]
    )
    llm = _make_llm()
    response = (prompt | llm).invoke(
        {"question": question, "context": context, "answer": answer}
    )

    fallback = {
        "passed": False,
        "answers_question": False,
        "answer_supported": False,
        "has_hallucination": True,
        "reason": "judge JSON 解析失败，保守判定为未通过",
        "suggested_action": "vector_search",
        "suggested_query": state.get("rewritten_query") or question,
    }
    result = parse_llm_json(response.content, fallback=fallback)
    answers_question = _as_bool(result.get("answers_question"))
    answer_supported = _as_bool(result.get("answer_supported"))
    has_hallucination = _as_bool(result.get("has_hallucination"))
    passed = (
        _as_bool(result.get("passed"))
        and answers_question
        and answer_supported
        and not has_hallucination
    )
    suggested_action = result.get("suggested_action") or "finish"
    if not passed and suggested_action == "finish":
        suggested_action = "vector_search"

    result = {
        "passed": passed,
        "answers_question": answers_question,
        "answer_supported": answer_supported,
        "has_hallucination": has_hallucination,
        "reason": result.get("reason") or "",
        "suggested_action": suggested_action,
        "suggested_query": result.get("suggested_query") or "",
    }

    retry_count = int(state.get("retry_count") or 0)
    if not result["passed"]:
        retry_count += 1

    print(f"[检查] passed={result['passed']} retry_count={retry_count}/{state.get('max_retries', config.AGENT_MAX_RETRIES)}")
    if result["reason"]:
        print(f"[检查] 原因: {result['reason']}")

    return {
        "judge_result": result,
        "retry_count": retry_count,
    }


def output_node(state: RAGState) -> dict:
    """
    节点5：输出节点。
    打印 LLM 回答，并附带检索来源引用。
    """
    answer = state.get("answer", "")
    sources = state.get("sources", [])
    judge_result = state.get("judge_result") or {}

    print("\n" + "=" * 60)
    print("回答:")
    print("-" * 60)
    print(answer)
    print("-" * 60)
    if judge_result and not judge_result.get("passed", True):
        print("检查提示:")
        print(f"  {judge_result.get('reason') or '答案未通过检查，但已达到重试上限或无法继续改进。'}")
    if sources:
        print("来源引用:")
        for s in sources:
            print(f"  {s}")
    print("=" * 60)

    return {"answer": answer, "sources": sources}
