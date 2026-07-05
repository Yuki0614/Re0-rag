from __future__ import annotations

"""
文档加载模块：使用 pymupdf4llm 将 PDF 转为 Markdown，保留表格、丢弃图片。
输出到 config.DOCS_DIR 目录。
"""

from pathlib import Path

import pymupdf4llm

import config


def pdf_to_markdown(pdf_path: str | Path, output_dir: str | Path | None = None) -> str:
    """
    将 PDF 文件转换为 Markdown 文件。

    Args:
        pdf_path:  PDF 文件路径
        output_dir: 输出目录，默认为 config.DOCS_DIR

    Returns:
        生成的 .md 文件路径

    转换策略：
        - 表格：保留为 Markdown 表格
        - 图片：丢弃（write_images=False）
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir or config.DOCS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    if not pdf_path.suffix.lower() == ".pdf":
        raise ValueError(f"仅支持 PDF 文件，收到: {pdf_path.suffix}")

    # pymupdf4llm 将 PDF 转为 Markdown 文本
    md_text = pymupdf4llm.to_markdown(
        doc=str(pdf_path),
        write_images=False,       # 丢弃图片
        page_chunks=False,        # 不分页，输出完整文档
    )

    # 输出文件与 PDF 同名，后缀改为 .md
    output_path = output_dir / f"{pdf_path.stem}.md"
    output_path.write_text(md_text, encoding="utf-8")

    return str(output_path)
