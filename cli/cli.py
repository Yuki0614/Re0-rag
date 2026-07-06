from __future__ import annotations

"""
re0-rag 命令行界面：文档管理（导入 / 删除 / 列表）与问答。
命令的具体实现在此，main.py 负责解析 argv 并分发到这里。
"""

import contextlib
import io
import shutil
import sys
import uuid
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config

# 本地文件目录（统一从根 config 读取）
DOCS_DIR = config.DOCS_DIR
CHUNKS_DIR = config.CHUNKS_DIR
META_DIR = config.META_DIR


def _reconfigure_stdout_utf8() -> None:
    """Windows 控制台默认 GBK，强制 stdout 用 UTF-8 避免中文乱码。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _print_quiet_answer(result: dict) -> None:
    """安静模式只输出最终答案，不展示中间检索流程。"""
    answer = (result.get("answer") or "").strip()
    if answer:
        print(answer)
        return
    judge_result = result.get("judge_result") or {}
    reason = judge_result.get("reason")
    print(reason or "未生成答案。")


def _import_pdf(pdf_path: Path) -> None:
    """导入单个 PDF：PDF → Markdown → chunks → embeddings → Qdrant。"""
    from db.chunk import save_split, split_markdown
    from db.embedding import embed_chunks, embed_tables, get_embedding_model
    from db.loader import pdf_to_markdown
    from db.meta import extract_metadata, save_metadata
    from db.manager import delete_by_source, insert_chunks, insert_tables

    print(f"[1/5] PDF → Markdown: {pdf_path.name}")
    md_path = pdf_to_markdown(pdf_path, DOCS_DIR)
    print(f"      输出: {md_path}")

    print("[2/5] 论文元数据抽取（LLM）")
    meta = extract_metadata(pdf_path, md_path)
    meta_path = save_metadata(meta, META_DIR / f"{pdf_path.stem}_meta.json")
    print(f"      标题: {meta.get('title') or '(未识别)'}")
    print(f"      期刊: {meta.get('journal') or '(未识别)'}")
    print(f"      作者: {', '.join(meta.get('authors') or []) or '(未识别)'}")
    print(f"      已保存: {meta_path}")

    print("[3/5] Markdown → Tables / Parent-Child Chunks")
    # split_data: {tables, parents, children}
    split_data = split_markdown(md_path, pdf_path=pdf_path)
    tables = split_data["tables"]
    parents = split_data["parents"]
    children = split_data["children"]
    for warning in split_data.get("warnings") or []:
        print(f"      [表格增强] {warning}")

    # 把抽取出的论文元数据注入每个 chunk 的 metadata,随向量一起入 Qdrant,
    # 检索命中时即可携带所属论文的标题/摘要,供生成层引用与增强。
    # parents 同样注入,以便召回时 format_context 能取到论文四元信息。
    meta_fields = {
        "title": meta.get("title"),
        "authors": meta.get("authors"),
        "journal": meta.get("journal"),
        "abstract": meta.get("abstract"),
    }
    for plist in (parents, children):
        for c in plist:
            c["metadata"].update(meta_fields)
    for t in tables:
        t["metadata"].update(meta_fields)
        title = meta.get("title")
        if title and f"Paper: {title}" not in (t.get("index_text") or ""):
            t["index_text"] = f"Paper: {title}\n" + (t.get("index_text") or "")

    paper_dir = CHUNKS_DIR / pdf_path.stem
    save_split(
        {"tables": tables, "parents": parents, "children": children},
        paper_dir,
        pdf_path.stem,
    )
    print(f"      表格 {len(tables)} 张 / parent {len(parents)} 块 / child {len(children)} 块")
    print(f"      已保存: {paper_dir}")

    print("[4/5] Children / Tables → Embeddings (all-MiniLM-L6-v2)")
    model = get_embedding_model()
    children = embed_chunks(children, model)
    tables_for_index = embed_tables(tables, model) if tables else []
    print(f"      向量化完成，维度: {len(children[0]['embedding']) if children else 0}")

    print("[5/5] Children / Tables 存入 Qdrant（替换同名文档旧索引；parents 仅落盘不入库）")
    delete_by_source(f"{pdf_path.stem}.md")
    count = insert_chunks(children) if children else 0
    table_count = insert_tables(tables_for_index) if tables_for_index else 0
    print(f"      已写入 {count} 条 child 记录")
    print(f"      已写入 {table_count} 条 table 记录")

    print(f"\n完成！文档 '{pdf_path.name}' 已入库。")


def cmd_import(args: list[str]) -> int:
    """
    导入文档：PDF → Markdown → 切分 → 向量化 → 替换写入向量库。
    用法: python main.py -cli import <PDF路径或目录路径>
    """
    _reconfigure_stdout_utf8()

    if not args:
        print("用法: python main.py -cli import <PDF路径或目录路径>")
        return 1

    input_path = Path(" ".join(args))
    if not input_path.exists():
        print(f"[错误] 路径不存在: {input_path}")
        return 1

    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            print(f"[错误] 仅支持 PDF 文件或目录，收到: {input_path.suffix or input_path.name}")
            return 1
        _import_pdf(input_path)
        return 0

    if not input_path.is_dir():
        print(f"[错误] 不是可导入的文件或目录: {input_path}")
        return 1

    entries = sorted(input_path.iterdir(), key=lambda p: p.name.lower())
    pdf_paths = []
    skipped = []
    for path in entries:
        if path.is_file() and path.suffix.lower() == ".pdf":
            pdf_paths.append(path)
        else:
            skipped.append(path)

    print(f"[批量导入] 目录: {input_path}")
    print(f"[批量导入] 发现 {len(pdf_paths)} 个 PDF，跳过 {len(skipped)} 个非 PDF 项")
    for path in skipped:
        print(f"  - 跳过: {path.name}")

    if not pdf_paths:
        print("[批量导入] 没有可导入的 PDF。")
        return 1

    succeeded = []
    failed = []
    for index, pdf_path in enumerate(pdf_paths, start=1):
        print(f"\n========== [{index}/{len(pdf_paths)}] {pdf_path.name} ==========")
        try:
            _import_pdf(pdf_path)
            succeeded.append(pdf_path)
        except Exception as e:
            failed.append((pdf_path, e))
            print(f"[错误] 导入失败: {pdf_path.name}: {e}")

    print("\n[批量导入] 汇总")
    print(f"  成功: {len(succeeded)}")
    print(f"  失败: {len(failed)}")
    print(f"  跳过: {len(skipped)}")
    if failed:
        print("  失败文件:")
        for path, err in failed:
            print(f"    - {path.name}: {err}")
    return 1 if failed else 0


def cmd_delete(args: list[str]) -> int:
    """
    删除文档：从 Qdrant 删除该文档所有 child/table,并清理本地 .md / chunks目录 / meta.json。
    用法: python main.py -cli delete <文档名或source>
          source 即 list 命令显示的文件名（如 xxx.md）。
    """
    _reconfigure_stdout_utf8()
    from db.manager import delete_by_source, list_sources

    if not args:
        print("用法: python main.py -cli delete <文档名>")
        print("      文档名可用 `python main.py -cli list` 查看")
        return 1

    source = " ".join(args).strip()

    # 先确认向量库里确实有这个文档
    existing = {item["source"] for item in list_sources()}
    if source not in existing:
        print(f"[错误] 向量库中未找到文档: {source}")
        if existing:
            print("现有文档:")
            for s in sorted(existing):
                print(f"  - {s}")
        return 1

    # 1) 删除 Qdrant 记录
    delete_by_source(source)
    print(f"[1/2] 已从向量库删除: {source}")

    # 2) 清理本地文件（.md / chunks 目录 / meta.json）
    md_file = DOCS_DIR / source
    chunks_dir = CHUNKS_DIR / Path(source).stem
    meta_file = META_DIR / f"{Path(source).stem}_meta.json"
    removed = []
    if md_file.exists():
        md_file.unlink()
        removed.append(str(md_file))
    if chunks_dir.exists():
        shutil.rmtree(chunks_dir)
        removed.append(str(chunks_dir))
    if meta_file.exists():
        meta_file.unlink()
        removed.append(str(meta_file))
    if removed:
        print("[2/2] 已清理本地文件:")
        for f in removed:
            print(f"  - {f}")
    else:
        print("[2/2] 未找到对应本地文件（可能已删除）")

    print(f"\n完成！文档 '{source}' 已彻底删除。")
    return 0


def cmd_list(args: list[str]) -> int:
    """
    列出向量库中所有已索引文档及其 chunk 数。
    用法: python main.py -cli list
    """
    _reconfigure_stdout_utf8()
    from db.manager import list_sources
    from db.meta import load_metadata

    sources = list_sources()
    if not sources:
        print("向量库为空，暂无文档。")
        print("可用 `python main.py -cli import <PDF路径或目录路径>` 导入。")
        return 0

    total = sum(item["chunks"] for item in sources)
    print(f"共 {len(sources)} 个文档，{total} 个片段：")
    print(f"{'序号':<4}{'标题 / 作者':<64}{'期刊':<24}{'片段数':<8}")
    print("-" * 100)
    for i, item in enumerate(sources, start=1):
        meta = load_metadata(item["source"]) or {}
        title = meta.get("title") or item["source"]
        authors = ", ".join((meta.get("authors") or [])[:3])
        journal = meta.get("journal") or "-"
        label = f"{title}" + (f" / {authors}" if authors else "")
        # 截断超长标签，避免错列
        if len(label) > 62:
            label = label[:60] + ".."
        print(f"{i:<4}{label:<64}{journal[:22]:<24}{item['chunks']:<8}")
    return 0


def cmd_reindex_keywords(args: list[str]) -> int:
    """Rebuild the standalone Qdrant BM25 collection from saved chunks."""
    _reconfigure_stdout_utf8()
    from db.manager import rebuild_keyword_collection_from_chunks

    print("Rebuilding Qdrant BM25 keyword index from db/chunks ...")
    result = rebuild_keyword_collection_from_chunks()
    print(f"Done: {result['children']} child records, {result['tables']} table records.")
    return 0


def cmd_query(args: list[str], trace: bool = False) -> int:
    """
    RAG 问答：输入问题，走 input→rewrite→route→tool→llm→judge→output 链路。
    用法: python main.py -cli query <问题>      单次提问后退出（独立会话）
          python main.py -cli query -t <问题>   单次提问并显示完整流程
          python main.py -cli query             交互式循环提问（共享会话历史，输入 exit/quit 退出）
    """
    _reconfigure_stdout_utf8()
    from re0rag.graph import preload as preload_rag
    from re0rag.graph import run as run_rag

    if args and args[0] in ("-t", "--trace"):
        trace = True
        args = args[1:]

    question = " ".join(args).strip()
    if question:
        # 带参数 → 单次提问，用独立 thread_id（不累积历史）
        result = run_rag(question, thread_id=str(uuid.uuid4()), verbose=trace)
        if not trace:
            _print_quiet_answer(result)
        return 0

    # 无参数 → 交互式循环：启动时预加载模型，之后反复提问
    # 同一 thread_id 共享对话历史，实现多轮记忆
    session_id = "console"
    print("系统启动中...")
    if trace:
        preload_rag()
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            preload_rag()
    print("系统就绪，输入问题开始提问（输入 exit 或 quit 退出）。")
    if trace:
        print("当前为 trace 模式，将显示改写、路由、工具调用和检查流程。")
    print("-" * 60)

    while True:
        try:
            question = input("\n请输入问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            return 0
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q", "退出"):
            print("再见！")
            return 0
        try:
            result = run_rag(question, thread_id=session_id, verbose=trace)
            if not trace:
                _print_quiet_answer(result)
        except Exception as e:
            print(f"[错误] 问答失败: {e}")
            print("可继续提问，或输入 exit 退出。")
    return 0


# 子命令分发表
COMMANDS = {
    "import": cmd_import,
    "delete": cmd_delete,
    "list": cmd_list,
    "reindex-keywords": cmd_reindex_keywords,
    "query": cmd_query,
}


def main(argv: list[str] | None = None) -> int:
    """
    CLI 主入口：解析 argv 分发到对应命令。
    无子命令时默认进入问答（python main.py -cli "问题"）。

    Returns:
        进程退出码
    """
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv:
        # 无参数 → 交互式问答
        return cmd_query([])

    if argv[0] in ("-t", "--trace"):
        # trace 模式快捷入口：python main.py -t [问题]
        return cmd_query(argv[1:], trace=True)

    cmd = argv[0]
    rest = argv[1:]

    if cmd in COMMANDS:
        if cmd == "query":
            return cmd_query(rest)
        return COMMANDS[cmd](rest)

    # 不是已知命令 → 当作问题处理（python main.py -cli "问题"）
    return cmd_query(argv)


if __name__ == "__main__":
    raise SystemExit(main())
