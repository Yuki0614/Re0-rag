"""
RAG 查询生成图：组装 Agentic RAG StateGraph，编译并提供 run 入口。
流程：input -> rewrite_query -> route -> tool -> llm -> judge -> output
judge 不通过且未超过重试上限时，回到 route 重新选择 tool。
多轮对话：编译时挂 MemorySaver 检查点，用 thread_id 区分会话，
state（含 messages 历史）跨 invoke 自动保留。
"""

import contextlib
import io

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from .edges import add_edges
from .state import RAGState

# 编译后的图实例缓存，避免每次提问都重新编译
_compiled_app = None


def build_graph():
    """
    构建 RAG StateGraph 并编译（挂 MemorySaver 检查点）。

    Returns:
        编译后的可执行图（CompiledGraph），带 checkpointer
    """
    graph = StateGraph(RAGState)
    add_edges(graph)
    # MemorySaver：内存级检查点，state 按 thread_id 跨 invoke 保留
    return graph.compile(checkpointer=MemorySaver())


def get_app():
    """
    获取（并缓存）编译后的图实例。首次调用时编译，之后复用。

    Returns:
        编译后的可执行图（CompiledGraph）
    """
    global _compiled_app
    if _compiled_app is None:
        _compiled_app = build_graph()
    return _compiled_app


def preload() -> None:
    """
    预加载：编译图并触发 embedding 模型加载。
    在系统启动时调用，避免首次提问才加载造成的延迟。
    """
    # 编译图
    get_app()
    # 触发 embedding 模型加载（tool 层会复用同一实例）
    from db.embedding import get_embedding_model
    get_embedding_model()


def run(question: str, thread_id: str = "default", verbose: bool = True) -> dict:
    """
    单次运行 RAG 链路：输入问题，返回最终状态。
    复用已编译的图实例，不重复编译。
    通过 thread_id 区分会话，同一 thread_id 下的多次调用共享对话历史。

    Args:
        question:   用户提问原文
        thread_id:  会话标识，同一 thread_id 下 messages 历史累积
        verbose:    是否打印 input/rewrite/route/tool/judge/output 全流程

    Returns:
        最终状态 dict，含 question / documents / sources / answer / messages 等
    """
    app = get_app()
    config_run = {"configurable": {"thread_id": thread_id}}
    if verbose:
        result = app.invoke({"question": question}, config=config_run)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            result = app.invoke({"question": question}, config=config_run)
    return result


if __name__ == "__main__":
    # 直接 python -m re0rag.graph 时交互式提问
    q = input("请输入问题: ").strip()
    if q:
        run(q)
