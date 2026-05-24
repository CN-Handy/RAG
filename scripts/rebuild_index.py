"""
重建 Elasticsearch 向量索引

适用场景：
  - 修改了 chunk_size 或 chunk_overlap 后需要重新切分并索引
  - 更换了 embedding 模型（dims 变化时需先删除索引、修改 config.yaml，再运行本脚本）
  - ES 数据意外丢失需要从原始文件恢复

用法（从项目根目录运行）：
  python scripts/rebuild_index.py                   # 重建全部文档
  python scripts/rebuild_index.py --kb-id 1         # 仅重建 knowledge_id=1
  python scripts/rebuild_index.py --dry-run         # 预览待操作文档，不执行
  python scripts/rebuild_index.py --kb-id 1 --dry-run
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import config
from app.core.rag import RAG
from app.db.models import KnowledgeDocument, Session
from app.db.vector_store import es, init_es

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _clear_es_data(kb_id: int | None) -> None:
    """清除 ES 中的旧向量数据。全量重建时删除并重建索引；按知识库时只删该 KB 的数据。"""
    if kb_id is None:
        logger.info("删除并重建全部 ES 索引…")
        for idx in ("chunk_info", "document_meta"):
            if es.indices.exists(index=idx):
                es.indices.delete(index=idx)
                logger.info(f"  索引 {idx!r} 已删除")
        init_es()
    else:
        logger.info(f"清除 knowledge_id={kb_id} 的 ES 数据…")
        for idx in ("chunk_info", "document_meta"):
            resp = es.delete_by_query(
                index=idx,
                body={"query": {"term": {"knowledge_id": kb_id}}},
                refresh=True,
            )
            deleted = resp.get("deleted", 0)
            logger.info(f"  {idx!r}: 删除 {deleted} 条")


def _load_docs(kb_id: int | None) -> list:
    """从 SQLite 加载待重建的文档列表（排除已标记为 failed 的）。"""
    with Session() as session:
        q = session.query(KnowledgeDocument)
        if kb_id is not None:
            q = q.filter(KnowledgeDocument.knowledge_id == kb_id)
        q = q.filter(KnowledgeDocument.parse_status != "failed")
        # expire_on_commit=False 不在 session 内，detach 后属性仍可访问
        docs = q.all()
        # 在 session 关闭前把需要的属性读出来
        result = [
            {
                "document_id": d.document_id,
                "knowledge_id": d.knowledge_id,
                "title": str(d.title),
                "file_type": str(d.file_type),
                "file_path": str(d.file_path),
                "parse_status": str(d.parse_status),
            }
            for d in docs
        ]
    return result


def rebuild(kb_id: int | None, dry_run: bool) -> None:
    if not es.ping():
        logger.error("无法连接 Elasticsearch，请先确认服务已启动")
        sys.exit(1)

    docs = _load_docs(kb_id)
    if not docs:
        logger.info("未找到符合条件的文档，无需重建")
        return

    logger.info(f"待重建文档（共 {len(docs)} 份）：")
    for d in docs:
        flag = "⚠️  文件缺失" if not Path(d["file_path"]).exists() else ""
        logger.info(
            f"  [{d['document_id']}] {d['title']}"
            f"  kb={d['knowledge_id']}  status={d['parse_status']}  {flag}"
        )

    if dry_run:
        logger.info("--dry-run 模式，不执行实际操作，退出")
        return

    # ── 清除旧数据 ────────────────────────────────────────────────────────────
    _clear_es_data(kb_id)

    # ── 逐一重建 ──────────────────────────────────────────────────────────────
    rag = RAG()
    ok, fail = 0, 0

    for i, d in enumerate(docs, 1):
        doc_id = d["document_id"]
        file_path = d["file_path"]
        logger.info(f"[{i}/{len(docs)}] 重建 document_id={doc_id} [{d['title']}]…")

        if not Path(file_path).exists():
            logger.warning(f"  原始文件不存在，跳过: {file_path}")
            rag._update_parse_status(doc_id, "failed")
            fail += 1
            continue

        # 标记进行中，方便监控
        rag._update_parse_status(doc_id, "processing")
        try:
            # extract_content 内部会在完成后调用 _update_parse_status("completed"/"failed")
            rag.extract_content(
                knowledge_id=d["knowledge_id"],
                document_id=doc_id,
                title=d["title"],
                file_type=d["file_type"],
                file_path=file_path,
            )
            ok += 1
            logger.info(f"  ✓ 完成")
        except Exception as e:
            logger.error(f"  ✗ 异常: {e}", exc_info=True)
            rag._update_parse_status(doc_id, "failed")
            fail += 1

    logger.info("=" * 50)
    logger.info(f"重建完成：成功 {ok}，失败 {fail}，共 {len(docs)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="重建 Elasticsearch 向量索引",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--kb-id", type=int, default=None,
        metavar="ID",
        help="仅重建指定知识库（不传则重建全部）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅打印将要操作的文档，不执行实际重建",
    )
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("Advanced RAG — 索引重建")
    logger.info(f"  chunk_size    = {config['rag']['chunk_size']}")
    logger.info(f"  chunk_overlap = {config['rag']['chunk_overlap']}")
    logger.info(f"  embedding     = {config['rag']['embedding_model']}")
    logger.info(f"  target        = {'全部知识库' if args.kb_id is None else f'knowledge_id={args.kb_id}'}")
    if args.dry_run:
        logger.info("  mode          = dry-run（不执行写操作）")
    logger.info("=" * 50)

    rebuild(kb_id=args.kb_id, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
