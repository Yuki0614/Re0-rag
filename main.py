from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def _run_cli(argv: list[str]) -> int:
    if argv and argv[0] in ("-h", "--help"):
        print("usage: python main.py -cli [command] [args]")
        print("")
        print("commands:")
        print("  import <PDF path or directory path>")
        print("  reindex-keywords")
        print("  reindex-graph")
        print("  delete <source.md>")
        print("  list")
        print("  query [-t] <question>")
        print("")
        print("retrieval:")
        print("  vector / BM25 初筛 → BCE cross-encoder Rerank 精排 → 父章节与表格证据")
        print("  Set RE0RAG_RERANK_ENABLED=0 to disable reranking.")
        print("")
        print("trace:")
        print("  python main.py -t [question]")
        return 0

    from cli.cli import main as cli_main

    return cli_main(argv)


def _is_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) != 0


def _find_free_port(host: str, preferred_port: int) -> int:
    for port in range(preferred_port, preferred_port + 20):
        if _is_port_free(host, port):
            return port
    return preferred_port


def _run_web(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Start the re0-rag web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--web-port", type=int, default=5173)
    parser.add_argument("--api-only", action="store_true")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ModuleNotFoundError:
        print("Missing dependency: uvicorn. Run `pip install -r requirements.txt`.")
        return 1

    if not args.api_only:
        web_port = _find_free_port(args.host, args.web_port)
        if web_port != args.web_port:
            print(f"[web] Port {args.web_port} is in use, using {web_port}.")
        os.environ["RE0RAG_START_VITE"] = "1"
        os.environ["RE0RAG_WEB_HOST"] = args.host
        os.environ["RE0RAG_WEB_PORT"] = str(web_port)
    else:
        os.environ["RE0RAG_START_VITE"] = "0"
    print(f"[api] Backend:  http://{args.host}:{args.port}")

    uvicorn.run(
        "api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        app_dir=str(PROJECT_ROOT),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-cli", "--cli"):
        return _run_cli(argv[1:])
    if argv and argv[0] in ("-t", "--trace"):
        return _run_cli(argv)
    return _run_web(argv)


if __name__ == "__main__":
    raise SystemExit(main())
