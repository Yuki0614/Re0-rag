"""
RAG 状态定义：langgraph 各节点间传递的共享状态。
继承 MessagesState，自带 messages 字段（add_messages 追加 reducer），
用于多轮对话记忆。配合 graph 的 MemorySaver 检查点，state 跨 invoke 保留。
"""

from langgraph.graph import MessagesState


class RAGState(MessagesState):
    """
    RAG 链路的共享状态，贯穿五个节点。

    字段流转：
        input_node      -> question, retry_count, max_retries, messages(追加 HumanMessage)
        rewrite_node    -> rewritten_query
        route_node      -> route_decision, selected_tool, tool_query, route_history
        tool_node       -> tool_results, evidence, sources
        llm_node        -> answer, messages(追加 AIMessage)
        judge_node      -> judge_result, retry_count
        output_node     -> 输出并附带 sources / judge_result

    messages 字段由 MessagesState 提供，带 add_messages reducer（追加语义），
    累积全部对话历史，供 rewrite_node 与 llm_node 读取上文。
    """
    # 节点1 输入：用户提问原文
    question: str

    # 节点2 查询改写：消解指代后的独立检索 query
    rewritten_query: str

    # 节点3 路由：LLM 选择 tool，并给出 tool 查询
    route_decision: dict
    selected_tool: str
    tool_query: str
    route_history: list[dict]

    # 节点4 tool 执行：本地 tool 返回的证据与来源
    tool_results: list[dict]
    evidence: list[dict]    # 统一证据列表，通常为召回的 parent 全文
    sources: list[str]      # 来源描述，用于输出引用

    # 兼容旧检索字段：向量检索 tool 会继续写入这些字段，便于调试与脚本复用
    question_vector: list[float]
    documents: list[dict]   # child 命中 [{content, metadata, child_index, parent_id, parent_index, score}]
    parents: list[dict]     # 召回的 parent 全文 [{parent_id, content, parent_index, metadata, hit_child_indices}]

    # 节点5 生成：LLM 回答
    answer: str

    # 节点6 检查：judge 结构化结果与循环控制
    judge_result: dict
    retry_count: int
    max_retries: int
