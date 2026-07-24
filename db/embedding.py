from __future__ import annotations

"""
向量化模块：使用 embedding 模型将文本块转为向量。
模型名与缓存目录等超参数统一从根 config 读取。
模型下载由 db.hf_download 管理：优先官方 Hugging Face，失败后降级镜像。
"""

import sys
from pathlib import Path

# 把项目根加入 sys.path 以便 import 根 config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

from langchain_huggingface import HuggingFaceEmbeddings
from trace.telemetry import trace_span

from .hf_download import resolve_model_path


# 模型参数（来自根 config）
MODEL_NAME = config.EMBEDDING_MODEL_NAME
MODEL_CACHE_DIR = str(config.EMBEDDING_MODEL_DIR)

# 已加载的模型实例缓存，避免重复加载（加载一次约 1-2 秒）
_embedding_model = None


def _is_model_cached() -> bool:
    """
    检查模型是否已在本地缓存。
    HuggingFace 缓存在 model/embedding/ 下以 models-- 开头的目录中。
    """
    cache_dir = Path(MODEL_CACHE_DIR)
    if not cache_dir.exists():
        return False

    # 查找包含模型名的缓存目录
    for path in cache_dir.iterdir():
        if path.is_dir() and MODEL_NAME.replace("/", "--") in path.name:
            return True
    return False


@trace_span(
    "embedding.get_model",
    attributes=lambda args, kwargs, result: {
        "model": MODEL_NAME,
        "loaded": result is not None,
    },
)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    获取 Embedding 模型实例（单例缓存）。
    模型自动下载到 model/embedding/，已存在则直接加载。
    首次调用加载并缓存，后续调用直接返回同一实例，避免重复加载。

    Returns:
        HuggingFaceEmbeddings 实例
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    if _is_model_cached():
        print(f"[Embedding] 检测到模型: {MODEL_NAME}，直接加载")
    else:
        print(f"[Embedding] 未检测到本地模型: {MODEL_NAME}")

    model_path = resolve_model_path(MODEL_NAME, MODEL_CACHE_DIR)

    _embedding_model = HuggingFaceEmbeddings(
        model_name=model_path,
        cache_folder=MODEL_CACHE_DIR,
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )
    return _embedding_model


@trace_span(
    "embedding.embed_chunks",
    attributes=lambda args, kwargs, result: {
        "chunks.count": len(result or []),
        "vector.size": len((result or [{}])[0].get("embedding", [])) if result else 0,
    },
)
def embed_chunks(chunks: list[dict], model: HuggingFaceEmbeddings | None = None) -> list[dict]:
    """
    对 chunks 列表进行向量化，将向量附加到每个 chunk。

    Args:
        chunks: 包含 "content" 字段的文本块列表
        model:  Embedding 模型实例，为 None 时自动创建

    Returns:
        附加了 "embedding" 字段的 chunks 列表
    """
    if model is None:
        model = get_embedding_model()

    texts = [chunk["content"] for chunk in chunks]
    vectors = model.embed_documents(texts)

    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector

    return chunks


@trace_span(
    "embedding.embed_tables",
    attributes=lambda args, kwargs, result: {
        "tables.count": len(result or []),
        "vector.size": len((result or [{}])[0].get("embedding", [])) if result else 0,
    },
)
def embed_tables(tables: list[dict], model: HuggingFaceEmbeddings | None = None) -> list[dict]:
    """
    对表格 evidence 进行向量化。用于 embedding 的不是完整表体，而是 index_text：
    caption / columns / section context 等语义外壳；完整表体保留在 payload 供回答使用。
    """
    if model is None:
        model = get_embedding_model()
    if not tables:
        return tables

    texts = [t.get("index_text") or t.get("caption") or t.get("content", "") for t in tables]
    vectors = model.embed_documents(texts)
    for table, vector in zip(tables, vectors):
        table["embedding"] = vector
    return tables
