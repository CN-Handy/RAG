"""
查看当前知识库、文档解析状态和 ES 索引统计信息

运行前无需启动 FastAPI 服务，但需要 ES 可访问。

用法（从项目根目录运行）：
  python scripts/status.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import config
from app.db.models import KnowledgeDatabase, KnowledgeDocument, Session
from app.db.vector_store import es


def _es_count(index: str) -> str:
    try:
        n = es.count(index=index)["count"]
        return str(n)
    except Exception:
        return "N/A"


def _bar(count: int, total: int, width: int = 20) -> str:
    if total == 0:
        return " " * width
    filled = round(count / total * width)
    return "█" * filled + "░" * (width - filled)


def main() -> None:
    sep = "─" * 60

    # ── 服务配置摘要 ──────────────────────────────────────────────────────────
    print(sep)
    print("Advanced RAG — 状态概览")
    print(sep)
    print(f"  chunk_size    : {config['rag']['chunk_size']}")
    print(f"  chunk_overlap : {config['rag']['chunk_overlap']}")
    print(f"  embedding     : {config['rag']['embedding_model']}")
    print(f"  rerank        : {config['rag']['rerank_model']} (use={config['rag']['use_rerank']})")
    print(f"  llm_model     : {config['rag']['llm_model']}")
    print()

    # ── Elasticsearch ─────────────────────────────────────────────────────────
    print("Elasticsearch")
    print(sep)
    if not es.ping():
        print("  ✗ 无法连接（请确认 ES 服务已启动）")
    else:
        health = es.cluster.health()
        print(f"  状态      : {health.get('status', 'unknown')}")
        print(f"  chunk_info    : {_es_count('chunk_info')} 条向量片段")
        print(f"  document_meta : {_es_count('document_meta')} 条文档元数据")
    print()

    # ── 知识库 & 文档 ─────────────────────────────────────────────────────────
    print("知识库 / 文档")
    print(sep)
    with Session() as session:
        kbs = session.query(KnowledgeDatabase).order_by(KnowledgeDatabase.knowledge_id).all()

        if not kbs:
            print("  （暂无知识库）")
        else:
            for kb in kbs:
                docs = (
                    session.query(KnowledgeDocument)
                    .filter(KnowledgeDocument.knowledge_id == kb.knowledge_id)
                    .all()
                )
                total = len(docs)
                counts = {"completed": 0, "processing": 0, "pending": 0, "failed": 0}
                for d in docs:
                    status = str(d.parse_status)
                    counts[status] = counts.get(status, 0) + 1

                print(f"  [{kb.knowledge_id}] {kb.title}  ({kb.category})  — {total} 份文档")
                if total > 0:
                    for status, cnt in counts.items():
                        if cnt == 0:
                            continue
                        bar = _bar(cnt, total)
                        pct = cnt / total * 100
                        print(f"       {status:<12} {bar}  {cnt:>3} ({pct:5.1f}%)")

                # 列出解析失败的文档（方便排查）
                failed_docs = [d for d in docs if str(d.parse_status) == "failed"]
                if failed_docs:
                    print(f"       ⚠️  失败文档:")
                    for d in failed_docs:
                        exists = Path(str(d.file_path)).exists()
                        print(f"         - [{d.document_id}] {d.title}  文件{'存在' if exists else '缺失'}: {d.file_path}")
                print()

    print(sep)


if __name__ == "__main__":
    main()
