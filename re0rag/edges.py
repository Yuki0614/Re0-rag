"""
RAG 链路边定义。
Agentic RAG 流程：
input -> rewrite_query -> route -> tool -> graph_retrieval -> llm -> judge
judge 通过则 output；不通过且未超过重试上限则回到 route。
"""

import sys
from pathlib import Path

from langgraph.graph import END, START

# 把项目根加入 sys.path 以便 import 根 config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

from .nodes import (
    input_node,
    graph_retrieval_node,
    judge_node,
    llm_node,
    output_node,
    rewrite_node,
    route_node,
    summarize_memory_node,
    tool_node,
)


# 节点名称常量，供 graph.py 统一引用
NODE_INPUT = "input"
NODE_SUMMARIZE_MEMORY = "summarize_memory"
NODE_REWRITE = "rewrite_query"
NODE_ROUTE = "route"
NODE_TOOL = "tool"
NODE_GRAPH_RETRIEVAL = "graph_retrieval"
NODE_LLM = "llm"
NODE_JUDGE = "judge"
NODE_OUTPUT = "output"


def judge_router(state) -> str:
    """
    judge 后的条件路由。
    - passed=True：输出
    - passed=False 且 retry_count <= max_retries：回到 route 再尝试
    - 超过重试上限：输出当前答案，并由 output_node 打印检查提示
    """
    judge_result = state.get("judge_result") or {}
    if judge_result.get("passed"):
        return NODE_OUTPUT

    retry_count = int(state.get("retry_count") or 0)
    max_retries = int(state.get("max_retries") or config.AGENT_MAX_RETRIES)
    if retry_count <= max_retries:
        return NODE_ROUTE
    return NODE_OUTPUT


def add_edges(graph) -> None:
    """
    向 StateGraph 注册节点并连线。
    顺序：START -> input -> summarize_memory -> rewrite_query -> route -> tool -> graph_retrieval -> llm -> judge
          judge -> output / route

    Args:
        graph: langgraph 的 StateGraph 实例
    """
    graph.add_node(NODE_INPUT, input_node)
    graph.add_node(NODE_SUMMARIZE_MEMORY, summarize_memory_node)
    graph.add_node(NODE_REWRITE, rewrite_node)
    graph.add_node(NODE_ROUTE, route_node)
    graph.add_node(NODE_TOOL, tool_node)
    graph.add_node(NODE_GRAPH_RETRIEVAL, graph_retrieval_node)
    graph.add_node(NODE_LLM, llm_node)
    graph.add_node(NODE_JUDGE, judge_node)
    graph.add_node(NODE_OUTPUT, output_node)

    graph.add_edge(START, NODE_INPUT)
    graph.add_edge(NODE_INPUT, NODE_SUMMARIZE_MEMORY)
    graph.add_edge(NODE_SUMMARIZE_MEMORY, NODE_REWRITE)
    graph.add_edge(NODE_REWRITE, NODE_ROUTE)
    graph.add_edge(NODE_ROUTE, NODE_TOOL)
    graph.add_edge(NODE_TOOL, NODE_GRAPH_RETRIEVAL)
    graph.add_edge(NODE_GRAPH_RETRIEVAL, NODE_LLM)
    graph.add_edge(NODE_LLM, NODE_JUDGE)
    graph.add_conditional_edges(
        NODE_JUDGE,
        judge_router,
        {
            NODE_ROUTE: NODE_ROUTE,
            NODE_OUTPUT: NODE_OUTPUT,
        },
    )
    graph.add_edge(NODE_OUTPUT, END)
