from __future__ import annotations

"""Hugging Face 模型下载：本地缓存优先，官方源失败后使用镜像。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError


def _repo_id(model_name: str) -> str:
    """Expand Sentence Transformers' short model aliases to their Hugging Face repo ID."""
    return model_name if "/" in model_name else f"sentence-transformers/{model_name}"


def resolve_model_path(model_name: str, cache_dir: str | Path) -> str:
    """Return a local model snapshot, downloading from HF and then its mirror if needed."""
    cache_dir = str(cache_dir)
    repo_id = _repo_id(model_name)

    try:
        return snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        pass

    try:
        print(f"[HuggingFace] Downloading {repo_id} from {config.HF_PRIMARY_ENDPOINT} ...")
        return snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            endpoint=config.HF_PRIMARY_ENDPOINT,
        )
    except Exception as primary_error:
        print(
            f"[HuggingFace] Official download failed ({primary_error!s}); "
            f"retrying via mirror {config.HF_MIRROR_ENDPOINT} ..."
        )
        return snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            endpoint=config.HF_MIRROR_ENDPOINT,
        )
