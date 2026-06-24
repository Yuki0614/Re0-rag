"""
re0-rag 主入口：统一分发到各子命令。

用法:
    python main.py import <PDF路径或目录路径>  导入文档（PDF→MD→切分→向量化→入库）
    python main.py delete <文档名>       删除文档（向量库 + 本地文件）
    python main.py list                  列出已索引文档
    python main.py query <问题>          RAG 问答
    python main.py -t [问题]             RAG 问答并显示完整 trace 流程
    python main.py <问题>                等同于 query（向后兼容）
    python main.py                       交互式问答
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from ui.cli import main as cli_main


def main() -> None:
    raise SystemExit(cli_main())


if __name__ == "__main__":
    main()
