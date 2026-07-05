from __future__ import annotations

import contextlib
import io
import json
import os
import queue
import shutil
import subprocess
import time
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIST = PROJECT_ROOT / "web" / "dist"
UPLOAD_DIR = PROJECT_ROOT / "uploads"
WEB_HOST = os.getenv("RE0RAG_WEB_HOST", "127.0.0.1")
WEB_PORT = os.getenv("RE0RAG_WEB_PORT", "5173")

app = FastAPI(title="re0-rag API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        f"http://127.0.0.1:{WEB_PORT}",
        "http://localhost:5173",
        f"http://localhost:{WEB_PORT}",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.preload = {"status": "pending", "seconds": 0.0, "error": ""}
app.state.vite_process = None


class QueryRequest(BaseModel):
    question: str
    thread_id: Optional[str] = None
    trace: bool = False


class QueryResponse(BaseModel):
    answer: str
    thread_id: str
    sources: list[str]
    evidence: list[dict[str, Any]]
    documents: list[dict[str, Any]]
    route_history: list[dict[str, Any]]
    judge_result: dict[str, Any]


class SettingsPayload(BaseModel):
    base_url: str
    api_key: str
    model: str


def _preload_runtime() -> None:
    from re0rag.graph import preload

    preload()


def _npm_command() -> str | None:
    return shutil.which("npm.cmd") or shutil.which("npm")


def _start_vite_after_preload() -> subprocess.Popen | None:
    if os.getenv("RE0RAG_START_VITE", "0") != "1":
        return None

    web_dir = PROJECT_ROOT / "web"
    npm = _npm_command()
    if npm is None or not web_dir.exists():
        print("[web] Frontend is not available.")
        return None
    if not (web_dir / "node_modules").exists():
        print("[web] Frontend dependencies are not installed.")
        print("      Run: cd web && npm install")
        return None

    process = subprocess.Popen(
        [
            npm,
            "run",
            "dev",
            "--",
            "--host",
            WEB_HOST,
            "--port",
            WEB_PORT,
            "--strictPort",
        ],
        cwd=web_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    print(f"[web] Local:    http://{WEB_HOST}:{WEB_PORT}")
    return process


@app.on_event("startup")
async def preload_on_startup() -> None:
    started = time.perf_counter()
    print("[api] Preloading LangGraph and embedding model...")
    try:
        await run_in_threadpool(_preload_runtime)
    except Exception as exc:
        elapsed = time.perf_counter() - started
        app.state.preload = {
            "status": "failed",
            "seconds": round(elapsed, 2),
            "error": str(exc),
        }
        print(f"[api] Preload failed after {elapsed:.2f}s: {exc}")
        app.state.vite_process = _start_vite_after_preload()
        return

    elapsed = time.perf_counter() - started
    app.state.preload = {
        "status": "ready",
        "seconds": round(elapsed, 2),
        "error": "",
    }
    print(f"[api] Preload complete in {elapsed:.2f}s.")
    app.state.vite_process = _start_vite_after_preload()


@app.on_event("shutdown")
async def shutdown_frontend() -> None:
    process = app.state.vite_process
    if process and process.poll() is None:
        process.terminate()


def _safe_name(name: str) -> str:
    clean = Path(name).name.strip()
    if not clean:
        clean = f"upload-{uuid.uuid4().hex}.pdf"
    return "".join(ch for ch in clean if ch not in '<>:"/\\|?*')


def _settings_payload() -> dict[str, str]:
    return {
        "base_url": config.LLM_BASE_URL or "",
        "api_key": config.LLM_API_KEY or "",
        "model": config.LLM_MODEL or "",
    }


def _write_env_settings(payload: SettingsPayload) -> None:
    env_path = PROJECT_ROOT / ".env"
    updates = {
        "RE0RAG_LLM_BASE_URL": payload.base_url.strip(),
        "RE0RAG_LLM_API_KEY": payload.api_key.strip(),
        "RE0RAG_LLM_MODEL": payload.model.strip(),
    }
    existing_lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    seen: set[str] = set()
    next_lines: list[str] = []

    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            next_lines.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key in updates:
            next_lines.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            next_lines.append(line)

    if next_lines and next_lines[-1].strip():
        next_lines.append("")
    for key, value in updates.items():
        if key not in seen:
            next_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(next_lines).rstrip() + "\n", encoding="utf-8")
    config.LLM_BASE_URL = updates["RE0RAG_LLM_BASE_URL"]
    config.LLM_API_KEY = updates["RE0RAG_LLM_API_KEY"]
    config.LLM_MODEL = updates["RE0RAG_LLM_MODEL"]


def _source_record(item: dict[str, Any]) -> dict[str, Any]:
    from db.meta import load_metadata

    meta = load_metadata(item["source"]) or {}
    return {
        "source": item["source"],
        "chunks": item.get("chunks", 0),
        "title": meta.get("title") or Path(item["source"]).stem,
        "authors": meta.get("authors") or [],
        "journal": meta.get("journal") or "",
        "abstract": meta.get("abstract") or "",
        "status": "ready",
    }


def _list_sources() -> list[dict[str, Any]]:
    from db.manager import list_sources

    return [_source_record(item) for item in list_sources()]


def _delete_source(source: str) -> None:
    from db.manager import delete_by_source, list_sources

    existing = {item["source"] for item in list_sources()}
    if source not in existing:
        raise ValueError(f"Source not found: {source}")

    delete_by_source(source)
    stem = Path(source).stem
    paths = [
        config.DOCS_DIR / source,
        config.CHUNKS_DIR / stem,
        config.META_DIR / f"{stem}_meta.json",
    ]
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _import_pdf_with_logs(pdf_path: Path) -> str:
    from cli.cli import _import_pdf

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        _import_pdf(pdf_path)
    return buffer.getvalue()


def _compact_evidence(items: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    result = []
    for item in items[:limit]:
        metadata = item.get("metadata") or {}
        content = item.get("content") or item.get("table_content") or ""
        result.append(
            {
                "content": content[:900],
                "source": metadata.get("source", ""),
                "title": metadata.get("title", ""),
                "page": metadata.get("page", metadata.get("page_number", "")),
                "score": item.get("score"),
                "doc_type": item.get("doc_type", "text"),
            }
        )
    return result


def _result_payload(result: dict[str, Any], thread_id: str) -> dict[str, Any]:
    return {
        "answer": (result.get("answer") or "").strip(),
        "thread_id": thread_id,
        "sources": result.get("sources") or [],
        "evidence": _compact_evidence(result.get("evidence") or []),
        "documents": _compact_evidence(result.get("documents") or []),
        "route_history": result.get("route_history") or [],
        "judge_result": result.get("judge_result") or {},
    }


def _progress_message(line: str) -> str | None:
    if "[输入]" in line:
        return "已收到问题"
    if "[改写]" in line:
        return "正在改写 query"
    if "[路由]" in line:
        return "正在选择检索策略"
    if "[工具]" in line:
        return "正在检索本地知识库"
    if "[生成]" in line:
        return "正在生成回答"
    if "[检查" in line:
        return "正在校验回答"
    if "来源引用" in line:
        return "正在整理引用来源"
    return None


class QueueWriter(io.TextIOBase):
    def __init__(self, events: "queue.Queue[dict[str, Any]]") -> None:
        self.events = events
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit(line.strip())
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            self._emit(self._buffer.strip())
        self._buffer = ""

    def _emit(self, line: str) -> None:
        if not line:
            return
        status = _progress_message(line)
        if status:
            self.events.put({"type": "progress", "status": status, "detail": line})


def _sse(data: dict[str, Any]) -> str:
    return "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"


def _stream_query(payload: QueryRequest):
    question = payload.question.strip()
    thread_id = payload.thread_id or f"web-{uuid.uuid4().hex}"
    events: "queue.Queue[dict[str, Any] | None]" = queue.Queue()

    def worker() -> None:
        try:
            from re0rag.graph import run

            events.put({"type": "progress", "status": "正在改写 query", "detail": "启动 RAG 工作流"})
            writer = QueueWriter(events)
            with contextlib.redirect_stdout(writer):
                result = run(question, thread_id=thread_id, verbose=True)
            writer.flush()
            events.put({"type": "done", "result": _result_payload(result, thread_id)})
        except Exception as exc:
            events.put({"type": "error", "error": str(exc)})
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    while True:
        event = events.get()
        if event is None:
            break
        yield _sse(event)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "preload": app.state.preload}


@app.get("/api/sources")
async def sources() -> dict[str, Any]:
    try:
        items = await run_in_threadpool(_list_sources)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"items": items, "total": len(items)}


@app.get("/api/settings")
def get_settings() -> dict[str, str]:
    return _settings_payload()


@app.post("/api/settings")
async def save_settings(payload: SettingsPayload) -> dict[str, str]:
    if not payload.base_url.strip() or not payload.api_key.strip() or not payload.model.strip():
        raise HTTPException(status_code=400, detail="Base URL, API Key, and Model are required.")
    try:
        await run_in_threadpool(_write_env_settings, payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _settings_payload()


@app.post("/api/query", response_model=QueryResponse)
async def query(payload: QueryRequest) -> QueryResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    thread_id = payload.thread_id or f"web-{uuid.uuid4().hex}"

    def _run() -> dict[str, Any]:
        from re0rag.graph import run

        return run(question, thread_id=thread_id, verbose=payload.trace)

    try:
        result = await run_in_threadpool(_run)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(**_result_payload(result, thread_id))


@app.post("/api/query-stream")
async def query_stream(payload: QueryRequest) -> StreamingResponse:
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")
    return StreamingResponse(_stream_query(payload), media_type="text/event-stream")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_DIR / _safe_name(file.filename)
    with target.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    try:
        logs = await run_in_threadpool(_import_pdf_with_logs, target)
        items = await run_in_threadpool(_list_sources)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"filename": target.name, "logs": logs, "sources": items}


@app.delete("/api/sources/{source:path}")
async def delete_source(source: str) -> dict[str, str]:
    try:
        await run_in_threadpool(_delete_source, source)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": "deleted", "source": source}


if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa_fallback(path: str) -> FileResponse:
        file_path = WEB_DIST / path
        if path and file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(WEB_DIST / "index.html")
else:

    @app.get("/")
    def dev_hint() -> dict[str, str]:
        return {
            "message": "API is running. Start the frontend with `cd web && npm run dev`.",
            "api": "/api/health",
        }
