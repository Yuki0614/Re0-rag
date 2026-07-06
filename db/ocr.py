from __future__ import annotations

"""
Optional OCR client for recovering table-heavy PDF layouts.

The importer treats OCR as a best-effort enhancement. If the service is not
configured or the request fails, callers receive a status/warning and can keep
the normal Markdown-based import path.
"""

import json
import time
from pathlib import Path
from typing import Any

import requests

import config


def ocr_available() -> bool:
    return bool(config.OCR_ENABLED and config.OCR_BASE_URL and config.OCR_API_KEY and config.OCR_MODEL)


def extract_pdf_markdown_pages(pdf_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    """
    Return OCR page Markdown for a PDF.

    Result shape:
        {
          "status": "ok|skipped|failed",
          "pages": [{"page_index": 0, "markdown": "..."}],
          "cache_path": "...",
          "warning": "..."
        }
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir or (config.OCR_DIR / pdf_path.stem))
    cache_path = output_dir / "pages.json"

    if not ocr_available():
        return {
            "status": "skipped",
            "pages": [],
            "warning": "OCR 未配置，已跳过 OCR 表格增强。",
        }

    if config.OCR_USE_CACHE and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cached["status"] = "ok"
            cached["cache_path"] = str(cache_path)
            cached["from_cache"] = True
            return cached
        except Exception:
            pass

    if not pdf_path.exists():
        return {
            "status": "failed",
            "pages": [],
            "warning": f"OCR 输入文件不存在: {pdf_path}",
        }

    try:
        result = _submit_and_poll(pdf_path)
        pages = _download_pages(result["json_url"])
    except Exception as exc:
        return {
            "status": "failed",
            "pages": [],
            "warning": f"OCR 调用失败，已跳过 OCR 表格增强: {exc}",
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_pdf": str(pdf_path),
        "model": config.OCR_MODEL,
        "job_id": result["job_id"],
        "pages": pages,
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    for page in pages:
        page_file = pages_dir / f"page_{page['page_index']:04d}.md"
        page_file.write_text(page.get("markdown", ""), encoding="utf-8")

    return {
        "status": "ok",
        "pages": pages,
        "cache_path": str(cache_path),
        "job_id": result["job_id"],
        "from_cache": False,
    }


def _submit_and_poll(pdf_path: Path) -> dict[str, str]:
    headers = {"Authorization": f"bearer {config.OCR_API_KEY}"}
    data = {
        "model": config.OCR_MODEL,
        "optionalPayload": json.dumps(config.OCR_OPTIONAL_PAYLOAD, ensure_ascii=False),
    }
    with pdf_path.open("rb") as handle:
        response = requests.post(
            config.OCR_BASE_URL,
            headers=headers,
            data=data,
            files={"file": handle},
            timeout=60,
        )
    response.raise_for_status()
    body = response.json()
    job_id = body.get("data", {}).get("jobId")
    if not job_id:
        raise RuntimeError(f"OCR 返回缺少 jobId: {body}")

    started = time.monotonic()
    while True:
        if time.monotonic() - started > config.OCR_TIMEOUT_SECONDS:
            raise TimeoutError(f"OCR 任务超时: {job_id}")

        poll = requests.get(f"{config.OCR_BASE_URL}/{job_id}", headers=headers, timeout=60)
        poll.raise_for_status()
        data = poll.json().get("data", {})
        state = data.get("state")
        if state == "done":
            json_url = data.get("resultUrl", {}).get("jsonUrl")
            if not json_url:
                raise RuntimeError(f"OCR 完成但缺少 jsonUrl: {data}")
            return {"job_id": job_id, "json_url": json_url}
        if state == "failed":
            raise RuntimeError(data.get("errorMsg") or f"OCR 任务失败: {job_id}")
        time.sleep(max(1, config.OCR_POLL_INTERVAL_SECONDS))


def _download_pages(json_url: str) -> list[dict[str, Any]]:
    response = requests.get(json_url, timeout=120)
    response.raise_for_status()
    pages: list[dict[str, Any]] = []
    page_index = 0
    for raw_line in response.text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        item = json.loads(line)
        result = item.get("result") or {}
        for parsed in result.get("layoutParsingResults") or []:
            markdown = ((parsed.get("markdown") or {}).get("text") or "").strip()
            pages.append(
                {
                    "page_index": page_index,
                    "markdown": markdown,
                }
            )
            page_index += 1
    return pages
