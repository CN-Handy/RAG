import uuid
import time
import datetime
import logging

from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile

from app.core.rag import RAG
from app.db.models import KnowledgeDatabase, KnowledgeDocument, Session
from app.db.vector_store import es
from app.schemas.api import DocumentResponse, DocumentItem, DocumentListResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(knowledge_id: int, token: str = "") -> DocumentListResponse:
    try:
        with Session() as session:
            records = session.query(KnowledgeDocument).filter(
                KnowledgeDocument.knowledge_id == knowledge_id
            ).order_by(KnowledgeDocument.create_dt.desc()).all()
        items = [
            DocumentItem(
                document_id=r.document_id,
                title=str(r.title),
                category=str(r.category),
                file_type=str(r.file_type),
                parse_status=str(r.parse_status),
            )
            for r in records
        ]
        return DocumentListResponse(documents=items, total=len(items), knowledge_id=knowledge_id)
    except Exception as e:
        logger.error(f"文档列表查询失败 knowledge_id={knowledge_id}: {e}", exc_info=True)
        return DocumentListResponse(documents=[], total=0, knowledge_id=knowledge_id)


@router.post("/document", response_model=DocumentResponse)
async def add_document(
    background_tasks: BackgroundTasks,
    knowledge_id: int = Form(...),
    title: str = Form(...),
    category: str = Form(...),
    file: UploadFile = File(...),
) -> DocumentResponse:
    start_time = time.time()
    try:
        with Session() as session:
            kb = session.query(KnowledgeDatabase).filter(
                KnowledgeDatabase.knowledge_id == knowledge_id
            ).first()
            if kb is None:
                return DocumentResponse(
                    request_id=str(uuid.uuid4()),
                    document_id=0, category="", title="",
                    knowledge_id=knowledge_id, file_type="",
                    parse_status="failed",
                    response_code=404, response_msg="知识库不存在，请先创建知识库",
                    process_status="failed", processing_time=time.time() - start_time,
                )

            record = KnowledgeDocument(
                title=title, category=category, knowledge_id=knowledge_id,
                file_path="", file_type=file.content_type,
                parse_status="pending",
                create_dt=datetime.datetime.now(), update_dt=datetime.datetime.now(),
            )
            session.add(record)
            session.flush()
            document_id = record.document_id
            session.commit()

        # 保存文件到本地
        file_path = f"upload_files/doc_{document_id}_{file.filename}"
        with open(file_path, "wb") as buf:
            buf.write(await file.read())

        # 更新文件路径，同时将解析状态置为 processing
        with Session() as session:
            doc = session.query(KnowledgeDocument).filter(
                KnowledgeDocument.document_id == document_id
            ).first()
            doc.file_path = file_path
            doc.parse_status = "processing"
            session.commit()

        # 后台异步提取文档内容（写入 ES），完成后更新 parse_status
        background_tasks.add_task(
            RAG().extract_content,
            knowledge_id=knowledge_id, document_id=document_id,
            title=title, file_type=file.content_type, file_path=file_path,
        )

        logger.info(f"文档添加成功 document_id={document_id}，后台解析中")
        return DocumentResponse(
            request_id=str(uuid.uuid4()),
            document_id=document_id, category=category, title=title,
            knowledge_id=knowledge_id, file_type=file.content_type,
            parse_status="processing",
            response_code=200, response_msg="文档添加成功，内容解析中",
            process_status="completed", processing_time=time.time() - start_time,
        )
    except Exception as e:
        logger.error(f"文档添加失败: {e}", exc_info=True)

    return DocumentResponse(
        request_id=str(uuid.uuid4()),
        document_id=0, category="", title="",
        knowledge_id=knowledge_id, file_type="",
        parse_status="failed",
        response_code=500, response_msg="文档添加失败",
        process_status="failed", processing_time=time.time() - start_time,
    )


@router.get("/document", response_model=DocumentResponse)
def get_document(document_id: int, token: str) -> DocumentResponse:
    start_time = time.time()
    try:
        with Session() as session:
            record = session.query(KnowledgeDocument).filter(
                KnowledgeDocument.document_id == document_id
            ).first()
        if record is not None:
            return DocumentResponse(
                request_id=str(uuid.uuid4()),
                document_id=document_id,
                category=str(record.category),
                title=str(record.title),
                knowledge_id=record.knowledge_id,
                file_type=str(record.file_type),
                parse_status=str(record.parse_status),
                response_code=200,
                response_msg="文档查询成功",
                process_status="completed",
                processing_time=time.time() - start_time,
            )
    except Exception as e:
        logger.error(f"文档查询失败 document_id={document_id}: {e}", exc_info=True)

    return DocumentResponse(
        request_id=str(uuid.uuid4()),
        document_id=document_id, category="", title="",
        knowledge_id=0, file_type="",
        parse_status="unknown",
        response_code=404, response_msg="文档不存在",
        process_status="completed", processing_time=time.time() - start_time,
    )


@router.delete("/document", response_model=DocumentResponse)
def delete_document(document_id: int, token: str) -> DocumentResponse:
    start_time = time.time()
    try:
        with Session() as session:
            record = session.query(KnowledgeDocument).filter(
                KnowledgeDocument.document_id == document_id
            ).first()
            if record is None:
                return DocumentResponse(
                    request_id=str(uuid.uuid4()),
                    document_id=document_id, category="", title="",
                    knowledge_id=0, file_type="",
                    parse_status="unknown",
                    response_code=404, response_msg="文档不存在",
                    process_status="completed", processing_time=time.time() - start_time,
                )

            kb_id = record.knowledge_id
            category = str(record.category)
            title = str(record.title)
            file_type = str(record.file_type)
            session.delete(record)
            session.commit()

        # 清理 ES 中的向量数据
        try:
            es.delete_by_query(
                index="chunk_info",
                body={"query": {"term": {"document_id": document_id}}},
            )
            es.delete_by_query(
                index="document_meta",
                body={"query": {"term": {"document_id": document_id}}},
            )
            logger.info(f"ES 数据清理完成 document_id={document_id}")
        except Exception as es_err:
            logger.warning(f"ES 清理失败（不影响删除结果）document_id={document_id}: {es_err}")

        return DocumentResponse(
            request_id=str(uuid.uuid4()),
            document_id=document_id, category=category, title=title,
            knowledge_id=kb_id, file_type=file_type,
            parse_status="deleted",
            response_code=200, response_msg="文档删除成功",
            process_status="completed", processing_time=time.time() - start_time,
        )
    except Exception as e:
        logger.error(f"文档删除失败 document_id={document_id}: {e}", exc_info=True)

    return DocumentResponse(
        request_id=str(uuid.uuid4()),
        document_id=document_id, category="", title="",
        knowledge_id=0, file_type="",
        parse_status="unknown",
        response_code=500, response_msg="文档删除失败",
        process_status="failed", processing_time=time.time() - start_time,
    )
