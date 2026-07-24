from __future__ import annotations

"""Cross-encoder reranker used after dense or BM25 candidate retrieval."""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

from .hf_download import resolve_model_path


_reranker: Any | None = None


def get_reranker() -> Any:
    """Load the locally cached reranker, downloading it on first use when needed."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        model_path = resolve_model_path(config.RERANK_MODEL_NAME, config.RERANK_MODEL_DIR)
        _reranker = CrossEncoder(model_path, max_length=config.RERANK_MAX_LENGTH)
    return _reranker


def rerank_hits(query: str, hits: list[dict], top_k: int) -> list[dict]:
    """Reorder retrieved child chunks by query-passage relevance and retain ``top_k``."""
    if not hits:
        return []
    if not config.RERANK_ENABLED:
        return hits[:top_k]

    scores = get_reranker().predict(
        [(query, hit.get("content", "")) for hit in hits],
        batch_size=config.RERANK_BATCH_SIZE,
        show_progress_bar=False,
    )
    ranked = []
    for hit, score in zip(hits, scores):
        record = dict(hit)
        record["retrieval_score"] = hit.get("score")
        record["rerank_score"] = float(score)
        record["score"] = float(score)
        ranked.append(record)
    return sorted(ranked, key=lambda hit: hit["rerank_score"], reverse=True)[:top_k]
