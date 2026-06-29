"""
re0-rag 全局配置：所有超参数集中在此。
db/ 与 re0rag/ 下的模块统一从根 config 读取，不在各自模块里硬编码。

修改任何超参数只需改本文件。
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# 路径配置（基于项目根，不依赖运行时 cwd）
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE lines from .env without overriding real env vars."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env(name: str, fallback_name: str | None = None, default: str = "") -> str:
    return os.getenv(name) or (os.getenv(fallback_name) if fallback_name else None) or default


_load_dotenv(PROJECT_ROOT / ".env")

DOCS_DIR = PROJECT_ROOT / "docs"                 # PDF 转出的 Markdown
CHUNKS_DIR = PROJECT_ROOT / "db" / "chunks"      # 切分后的 chunks JSON
VECTOR_DIR = PROJECT_ROOT / "db" / "vector"      # Qdrant 本地持久化目录
EMBEDDING_MODEL_DIR = PROJECT_ROOT / "model" / "embedding"  # embedding 模型缓存


# ──────────────────────────────────────────────
# 文档切分配置
# parent-child chunk：parent = 按固定长度硬切的粗粒度上下文（召回用），
# child = parent 内 500/100 切片（细粒度，检索用）。
# ──────────────────────────────────────────────
CHUNK_SIZE = 500          # child chunk 每段最大字符数（检索粒度）
CHUNK_OVERLAP = 100       # child 段间重叠字符数

PARENT_MAX_CHARS = 2000      # parent 每段最大字符数（防召回 context 爆 token）
PARENT_OVERLAP_CHARS = 200   # parent 段间重叠字符数

# db/chunks/ 下结构：{stem}/{stem}_tables.json / _parents.json / _children.json
# 表格单独存、不切分，并以 table evidence 写入向量库；parent/child 各一个聚合 JSON。


# ──────────────────────────────────────────────
# Embedding 模型配置
# ──────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_VECTOR_SIZE = 384   # all-MiniLM-L6-v2 输出 384 维


# ──────────────────────────────────────────────
# 向量库（Qdrant 本地模式）配置
# ──────────────────────────────────────────────
QDRANT_COLLECTION = "re0rag_docs"
QDRANT_KEYWORD_COLLECTION = "re0rag_docs_bm25"
QDRANT_BM25_VECTOR_NAME = "bm25"
QDRANT_BM25_MODEL = "Qdrant/bm25"


# ──────────────────────────────────────────────
# 检索配置
# ──────────────────────────────────────────────
RETRIEVE_TOP_K = 4      # 相似度检索返回的片段数量


# ──────────────────────────────────────────────
# Agentic RAG 配置
# 问答阶段由 route 节点选择本地 tool，再由 judge 节点检查答案；
# 若检查不通过，最多回到 route 节点重试 AGENT_MAX_RETRIES 次。
# ──────────────────────────────────────────────
AGENT_MAX_RETRIES = 2
AGENT_ALLOWED_TOOLS = ["vector_search", "keyword_search", "no_retrieval"]
VECTOR_TOP_K = RETRIEVE_TOP_K
KEYWORD_TOP_K = 4
TABLE_TOP_K = 3       # 表格 evidence 辅助召回数量；与正文 evidence 合并使用


# ──────────────────────────────────────────────
# LLM 配置（OpenAI 兼容接口）
# 从环境变量读取，避免把 API Key 写入源码。
# 注意：base_url 填到 /v1 为止，不要带 /chat/completions 后缀，
#       langchain 的 ChatOpenAI 会自动拼上 /chat/completions。
# ──────────────────────────────────────────────
LLM_BASE_URL = _env("RE0RAG_LLM_BASE_URL", "LLM_BASE_URL")
LLM_API_KEY = _env("RE0RAG_LLM_API_KEY", "LLM_API_KEY")
LLM_MODEL = _env("RE0RAG_LLM_MODEL", "LLM_MODEL")
LLM_TEMPERATURE = 0.3    # 生成温度，RAG 问答建议偏低
LLM_MAX_TOKENS = 1024    # 单次回答最大 token 数


# ──────────────────────────────────────────────
# Prompt 模板
# {question} 用户问题，{context} 检索到的片段拼接文本
# context 中已由 format_context 拼入所属论文的标题/期刊/作者/摘要，
# 因此 SYSTEM_PROMPT 引导 LLM 利用这些元信息并引用论文标题（而非文件名）。
# ──────────────────────────────────────────────
SYSTEM_PROMPT = (
    "你是一个严谨的文档问答助手。请仅根据下面提供的参考片段及其所属论文的标题与摘要回答用户问题。"
    "如果参考片段中没有相关信息，请直接说明无法回答，不要编造内容。"
    "回答时尽量引用所属论文的标题（而非文件名），并在合适处标注。"
)

USER_PROMPT_TEMPLATE = (
    "参考片段：\n{context}\n\n"
    "用户问题：{question}"
)


# ──────────────────────────────────────────────
# 查询改写 Prompt（多轮对话用）
# 把「历史对话 + 当前问题」改写成一个独立的检索 query，消解指代。
# {history} 对话历史，{question} 当前问题
# ──────────────────────────────────────────────
REWRITE_SYSTEM_PROMPT = (
    "你的任务是把用户当前的问题改写成一个独立、自包含的检索查询，"
    "用于在文档库中搜索相关信息。"
    "请根据对话历史消解问题中的指代（如“它”“这个”“上面提到的”），"
    "使其脱离上下文也能被理解。"
    "只输出改写后的查询，不要解释，不要加引号。"
    "如果当前问题已经自包含，直接原样输出。"
)

REWRITE_USER_PROMPT_TEMPLATE = (
    "对话历史：\n{history}\n\n"
    "当前问题：{question}\n\n"
    "改写后的检索查询："
)


# ──────────────────────────────────────────────
# Agentic RAG 路由 Prompt
# route 节点必须输出严格 JSON，供程序稳定选择 tool。
# ──────────────────────────────────────────────
ROUTE_SYSTEM_PROMPT = (
    "你是一个本地论文问答系统的检索路由器。你的任务是根据用户问题、改写后的检索 query、"
    "历史失败原因，选择下一步应该调用的本地 tool。只能使用以下三种 action：\n"
    "1. vector_search：适合语义解释、机制总结、方法比较、需要理解同义表达的问题。\n"
    "2. keyword_search：适合论文标题、模型名、数据集名、指标名、缩写、公式编号、表格编号等精确词匹配。\n"
    "3. no_retrieval：仅适合寒暄、系统操作、改写问题、与入库论文事实无关的问题。\n"
    "硬性规则：只要问题涉及论文内容、模型结构、实验结果、数据集、指标、结论或引用依据，就必须检索，不能选择 no_retrieval。\n"
    "如果上一轮检查失败，请根据失败原因换用更合适的 tool 或调整 query，避免重复无效检索。\n"
    "只输出 JSON，不要输出 markdown 代码块或解释文字。"
)

ROUTE_USER_PROMPT_TEMPLATE = (
    "用户原问题：\n{question}\n\n"
    "改写后的检索 query：\n{rewritten_query}\n\n"
    "上一轮检查结果/失败原因：\n{judge_feedback}\n\n"
    "已尝试过的 route：\n{route_history}\n\n"
    "请输出严格 JSON，格式如下：\n"
    '{{"action": "vector_search|keyword_search|no_retrieval", "query": "用于 tool 的查询文本", "reason": "选择原因"}}'
)


# ──────────────────────────────────────────────
# Agentic RAG 答案检查 Prompt
# judge 节点必须输出严格 JSON，供 graph 条件边判断是否重试。
# ──────────────────────────────────────────────
JUDGE_SYSTEM_PROMPT = (
    "你是一个严谨的 RAG 答案检查器。请根据用户问题、检索证据和候选答案判断："
    "答案是否回答了问题、是否被证据支持、是否出现证据中没有的事实或结论。"
    "如果证据不足以回答论文事实问题，应判定不通过。"
    "只输出 JSON，不要输出 markdown 代码块或解释文字。"
)

JUDGE_USER_PROMPT_TEMPLATE = (
    "用户问题：\n{question}\n\n"
    "检索证据：\n{context}\n\n"
    "候选答案：\n{answer}\n\n"
    "请输出严格 JSON，格式如下：\n"
    '{{"passed": true, "answers_question": true, "answer_supported": true, '
    '"has_hallucination": false, "reason": "判断理由", '
    '"suggested_action": "finish|vector_search|keyword_search", '
    '"suggested_query": "如果需要重试，给出新的查询；否则为空字符串"}}'
)


# ──────────────────────────────────────────────
# 论文元数据抽取配置
# 导入论文时用 LLM 从转好的 Markdown 抽取 标题/作者/期刊/摘要 四字段。
# LLM 复用上面的 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL，仅温度单独调低。
# ──────────────────────────────────────────────
META_DIR = PROJECT_ROOT / "db" / "meta"          # 元数据 JSON 目录（与 chunks 平级）
META_LLM_ENABLED = True                          # 是否启用 LLM 提取（关则四字段全为空）
META_LLM_HEAD_CHARS = 20000                      # 喂给 LLM 的 Markdown 前若干字符（覆盖摘要）
META_LLM_TEMPERATURE = 0.0                        # 抽取温度（低温降低幻觉）
META_LLM_MAX_TOKENS = 8192

# 抽取 prompt：{md_head} 是 pymupdf4llm 转出的 Markdown 前若干字符（结构清晰，
# 但部分双栏论文的 Abstract 段可能被丢失）；{fitz_page0} 是 fitz 直读的 PDF 前几页
# 原始文本（按物理顺序，含 Abstract 段但可能有少量乱序/换行）；{abstract_candidates}
# 是代码从 Markdown/PDF 文本中截出的疑似摘要窗口，用于提高稳定性。
META_LLM_SYSTEM_PROMPT = (
    "你是论文元数据抽取助手。系统给出两份同一论文的文本来源：一份 Markdown、一份 PDF 首页原样文本。"
    "系统还会给出若干“摘要候选片段”，这些片段来自 Abstract、A B S T R A C T、摘要等标记附近。"
    "综合这些证据抽取以下字段，返回严格 JSON：\n"
    '{{"title": str|null, "authors": [str]|null, "journal": str|null, "abstract": str|null}}\n'
    "字段说明：\n"
    "- title：论文完整标题（去掉任何 Markdown 标记如 \"## \"、去掉期刊名前缀，只留标题本身）。\n"
    "- authors：作者姓名列表（仅人名，去掉 \"Member, IEEE\" 等会员修饰）。若作者无逗号分隔，按姓名边界分别切分。\n"
    "- journal：期刊名或会议名。若论文无明确期刊/会议信息（如 arXiv 预印本页眉只有 \"arXiv:xxxx\"），返回 null，不要编造。\n"
    "- abstract：论文摘要原文。优先使用摘要候选片段或 PDF 首页/前几页文本中的 Abstract 段完整摘录；"
    "注意 Elsevier 论文可能写成 \"A B S T R A C T\"，IEEE 双栏论文可能在 Markdown 中丢失 Abstract 标题。"
    "若 Markdown 里 Introduction 标题先于 A B S T R A C T 出现，不要因此忽略后面的摘要候选。"
    "若无显式 Abstract，但标题/作者附近存在对论文整体工作的原文概述段，可作为 abstract。"
    "不要改写、不要翻译、不要总结。\n"
    "硬性要求：\n"
    "- 任何在文本中找不到依据的字段必须返回 null。禁止编造、禁止臆测。\n"
    "- 只输出 JSON，不要加 markdown 代码块，不要加任何解释。"
)

META_LLM_USER_PROMPT_TEMPLATE = (
    "摘要候选片段：\n{abstract_candidates}\n\n"
    "Markdown 片段：\n{md_head}\n\n"
    "PDF 首页文本：\n{fitz_page0}\n\n"
    "请输出 JSON："
)
