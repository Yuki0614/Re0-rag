"""
向量库管理模块：负责将向量 + 原文存入 Qdrant 向量数据库。
持久化路径、集合名、向量维度等超参数统一从根 config 读取。
"""

import re
import sys
import uuid
from collections import Counter
from pathlib import Path

# 把项目根加入 sys.path 以便 import 根 config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from db.chunk import load_split_file


# Qdrant 配置（来自根 config）
QDRANT_PATH = str(config.VECTOR_DIR)
COLLECTION_NAME = config.QDRANT_COLLECTION
VECTOR_SIZE = config.EMBEDDING_VECTOR_SIZE


def get_client() -> QdrantClient:
    """
    获取 Qdrant 本地客户端实例（持久化到 db/vector/）。
    """
    Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=QDRANT_PATH)


def ensure_collection(client: QdrantClient | None = None) -> QdrantClient:
    """
    确保集合存在，不存在则创建。

    Args:
        client: QdrantClient 实例，为 None 时自动创建

    Returns:
        QdrantClient 实例
    """
    if client is None:
        client = get_client()

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qdrant_models.VectorParams(
                size=VECTOR_SIZE,
                distance=qdrant_models.Distance.COSINE,
            ),
        )

    return client


def insert_chunks(
    chunks: list[dict],
    client: QdrantClient | None = None,
) -> int:
    """
    将带 embedding 的 child chunks 批量插入 Qdrant（upsert 幂等）。

    每个 chunk(实际是 child)需包含: content, embedding, metadata, child_index, parent_id, parent_index
    point_id 由 source + child_index 稳定生成(source::child::{i} 命名空间下的 uuid5),
    重复导入同一文档会覆盖而非新增。使用确定性 UUID v5 以满足 Qdrant 对整数/UUID 主键的要求。

    Args:
        chunks: 已向量化的 child 文本块列表
        client: QdrantClient 实例，为 None 时自动创建

    Returns:
        插入的点数量
    """
    client = ensure_collection(client)

    points = []
    for chunk in chunks:
        source = chunk.get("metadata", {}).get("source", "")
        point_id = _stable_point_id(source, chunk.get("child_index", 0))
        points.append(
            qdrant_models.PointStruct(
                id=point_id,
                vector=chunk["embedding"],
                payload={
                    "doc_type": "text",
                    "content": chunk["content"],
                    "metadata": chunk.get("metadata", {}),
                    "child_index": chunk.get("child_index", 0),
                    "parent_id": chunk.get("parent_id"),
                    "parent_index": chunk.get("parent_index", 0),
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


def insert_tables(
    tables: list[dict],
    client: QdrantClient | None = None,
) -> int:
    """
    将带 embedding 的 table evidence 批量插入 Qdrant。
    向量化文本是 table.index_text，payload 保留完整表格/上下文。
    """
    if not tables:
        return 0
    client = ensure_collection(client)

    points = []
    for table in tables:
        source = table.get("metadata", {}).get("source", "")
        table_index = table.get("table_index", 0)
        point_id = _stable_table_point_id(source, table_index)
        payload = {
            "doc_type": "table",
            "content": table.get("index_text") or table.get("caption") or table.get("content", ""),
            "table_content": table.get("content", ""),
            "caption": table.get("caption", ""),
            "columns": table.get("columns", []),
            "context_before": table.get("context_before", ""),
            "context_after": table.get("context_after", ""),
            "index_text": table.get("index_text", ""),
            "table_index": table_index,
            "table_id": table.get("table_id"),
            "metadata": table.get("metadata", {}),
            "parse_warnings": table.get("parse_warnings", []),
        }
        points.append(
            qdrant_models.PointStruct(
                id=point_id,
                vector=table["embedding"],
                payload=payload,
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


def _stable_point_id(source: str, child_index: int) -> str:
    """
    由 source + child_index 生成确定性 UUID v5 字符串（child 命名空间）。
    同一文档同一子片段多次导入得到相同 id，实现 upsert 幂等。
    """
    name = f"{source}::child::{child_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


def _stable_table_point_id(source: str, table_index: int) -> str:
    name = f"{source}::table::{table_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


def fetch_parents(
    hits: list[dict],
    client: QdrantClient | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    由检索命中的 child hits 召回对应的 parent 全文。

    parent 不入库(不向量化)，仅落盘 db/chunks/{stem}/{stem}_parents.json。
    这里按 hit 里 metadata.source 定位对应 papers 子目录,加载其 _parents.json,
    按 parent_id 去重取出对应 parent。

    Args:
        hits: search() 返回的 child 命中列表(每条含 parent_id, metadata.source)
        client: 仅用于接口一致性,未使用

    Returns:
        (parents, deduped_parent_ids)
        parents: [{parent_id, content, parent_index, metadata, hit_child_indices}]
                  按 hit 的出现顺序排序(去重)；hit_child_indices 为命中该 parent 的 child 索引
    """
    if not hits:
        return [], []

    chunks_root = config.CHUNKS_DIR
    # source -> 该 paper 的 parents.json 缓存(stem 加载一次)
    paper_parents_cache: dict[str, dict[str, dict]] = {}

    def _load_parents_index(source: str) -> dict[str, dict]:
        if source in paper_parents_cache:
            return paper_parents_cache[source]
        stem = Path(source).stem
        pjson = Path(chunks_root) / stem / f"{stem}_parents.json"
        result: dict[str, dict] = {}
        if pjson.exists():
            for p in load_split_file(pjson):
                result[p["parent_id"]] = p
        paper_parents_cache[source] = result
        return result

    parents_order: dict[str, dict] = {}
    seen: set[str] = set()
    for h in hits:
        source = h.get("metadata", {}).get("source", "")
        pid = h.get("parent_id")
        if not pid:
            continue
        idx = _load_parents_index(source)
        p = idx.get(pid)
        if p is None:
            continue
        if pid not in seen:
            seen.add(pid)
            rec = {
                "parent_id": pid,
                "content": p.get("content", ""),
                "parent_index": p.get("parent_index", 0),
                "metadata": p.get("metadata", {}),
                "hit_child_indices": [h.get("child_index")],
            }
            parents_order[pid] = rec
        else:
            parents_order[pid]["hit_child_indices"].append(h.get("child_index"))

    return list(parents_order.values()), list(parents_order.keys())


def fetch_linked_tables(parents: list[dict]) -> list[dict]:
    """根据 parent 中的 TABLE_REF 占位符加载关联表格 evidence。"""
    refs: list[tuple[str, int]] = []
    for parent in parents:
        source = parent.get("metadata", {}).get("source", "")
        if not source:
            continue
        for m in re.finditer(r"<TABLE_REF\s+id=(\d+)\b", parent.get("content", "")):
            refs.append((source, int(m.group(1))))
    return fetch_tables_by_refs(refs)


def fetch_tables_by_refs(refs: list[tuple[str, int]]) -> list[dict]:
    """从落盘 tables JSON 中按 (source, table_index) 读取表格。"""
    if not refs:
        return []
    seen = set()
    result = []
    cache: dict[str, list[dict]] = {}
    for source, table_index in refs:
        key = (source, table_index)
        if key in seen:
            continue
        seen.add(key)
        if source not in cache:
            stem = Path(source).stem
            pjson = Path(config.CHUNKS_DIR) / stem / f"{stem}_tables.json"
            cache[source] = load_split_file(pjson) if pjson.exists() else []
        for table in cache[source]:
            if table.get("table_index") == table_index:
                rec = dict(table)
                rec["doc_type"] = "table"
                result.append(rec)
                break
    return result


def list_sources(client: QdrantClient | None = None) -> list[dict]:
    """
    列出向量库中所有已索引文档（按 source 聚合）。

    Args:
        client: QdrantClient 实例，为 None 时自动创建

    Returns:
        [{"source": "xxx.md", "chunks": 12}, ...]，按 source 字母序排列
    """
    client = ensure_collection(client)

    # 用 payload 过滤拉取所有点的 source 字段
    count = client.count(COLLECTION_NAME, exact=True).count
    if count == 0:
        return []

    sources = []
    offset = 0
    batch = 256
    while True:
        result, _next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=batch,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in result:
            payload = point.payload or {}
            if payload.get("doc_type", "text") != "text":
                continue
            source = payload.get("metadata", {}).get("source")
            if source:
                sources.append(source)
        if _next_offset is None:
            break
        offset = _next_offset

    counter = Counter(sources)
    return [{"source": s, "chunks": n} for s, n in sorted(counter.items())]


def delete_by_source(source: str, client: QdrantClient | None = None) -> int:
    """
    按文档名(source)删除其在向量库中的所有 child/table point。

    Args:
        source: 文档名，对应 chunk metadata.source
        client: QdrantClient 实例，为 None 时自动创建

    Returns:
        删除的点数量（Qdrant 本地模式不返回精确删除数，此处返回 0 仅作占位）
    """
    client = ensure_collection(client)

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=qdrant_models.FilterSelector(
            filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="metadata.source",
                        match=qdrant_models.MatchValue(value=source),
                    )
                ]
            )
        ),
    )
    return 0

def search(
    query_vector: list[float],
    top_k: int = config.RETRIEVE_TOP_K,
    client: QdrantClient | None = None,
    doc_type: str | None = "text",
) -> list[dict]:
    """
    相似度检索，返回 top_k 个最相似的 chunk。

    Args:
        query_vector: 查询向量
        top_k:        返回条数
        client:       QdrantClient 实例

    Returns:
        [{"content": "...", "metadata": {...}, "score": 0.95}, ...]
    """
    client = ensure_collection(client)

    # qdrant-client >= 1.12 移除了 client.search，改用 query_points。
    # 返回 QueryResponse，结果在 .points 中，每个 point 含 .payload / .score。
    query_filter = None
    if doc_type:
        query_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="doc_type",
                    match=qdrant_models.MatchValue(value=doc_type),
                )
            ]
        )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
    )
    hits = response.points

    # 兼容旧索引：历史 child payload 没有 doc_type，text 检索空时回退无过滤。
    if doc_type == "text" and not hits:
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
        )
        hits = response.points

    return [
        {
            "content": hit.payload["content"],
            "doc_type": hit.payload.get("doc_type", "text"),
            "metadata": hit.payload.get("metadata", {}),
            "child_index": hit.payload.get("child_index", 0),
            "parent_id": hit.payload.get("parent_id"),
            "parent_index": hit.payload.get("parent_index", 0),
            "score": hit.score,
        }
        for hit in hits
    ]


def search_tables(
    query_vector: list[float],
    top_k: int = 3,
    client: QdrantClient | None = None,
) -> list[dict]:
    """检索 table evidence，返回完整表格 payload。"""
    client = ensure_collection(client)
    query_filter = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="doc_type",
                match=qdrant_models.MatchValue(value="table"),
            )
        ]
    )
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
    )
    return [
        {
            "doc_type": "table",
            "content": hit.payload.get("content", ""),
            "table_content": hit.payload.get("table_content", ""),
            "caption": hit.payload.get("caption", ""),
            "columns": hit.payload.get("columns", []),
            "context_before": hit.payload.get("context_before", ""),
            "context_after": hit.payload.get("context_after", ""),
            "table_index": hit.payload.get("table_index", 0),
            "table_id": hit.payload.get("table_id"),
            "metadata": hit.payload.get("metadata", {}),
            "parse_warnings": hit.payload.get("parse_warnings", []),
            "score": hit.score,
        }
        for hit in response.points
    ]
