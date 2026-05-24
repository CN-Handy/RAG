import uuid
import time
import datetime
import logging

from fastapi import APIRouter

from app.db.models import KnowledgeDatabase, Session
from app.schemas.api import KnowledgeRequest, KnowledgeResponse, KnowledgeItem, KnowledgeListResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/knowledge_bases", response_model=KnowledgeListResponse)
def list_knowledge_bases(token: str = "") -> KnowledgeListResponse:
    try:
        with Session() as session:
            records = session.query(KnowledgeDatabase).order_by(
                KnowledgeDatabase.create_dt.desc()
            ).all()
        items = [
            KnowledgeItem(
                knowledge_id=r.knowledge_id,
                title=str(r.title),
                category=str(r.category),
            )
            for r in records
        ]
        return KnowledgeListResponse(knowledge_bases=items, total=len(items))
    except Exception as e:
        logger.error(f"知识库列表查询失败: {e}", exc_info=True)
        return KnowledgeListResponse(knowledge_bases=[], total=0)


@router.post("/knowledge_base", response_model=KnowledgeResponse)
def add_knowledge_base(req: KnowledgeRequest) -> KnowledgeResponse:
    start_time = time.time()
    try:
        with Session() as session:
            record = KnowledgeDatabase(
                title=req.title,
                category=req.category,
                create_dt=datetime.datetime.now(),
                update_dt=datetime.datetime.now(),
            )
            session.add(record)
            session.flush()
            knowledge_id = record.knowledge_id
            session.commit()
        logger.info(f"知识库创建成功 knowledge_id={knowledge_id}")
        return KnowledgeResponse(
            request_id=str(uuid.uuid4()),
            knowledge_id=knowledge_id,
            category=req.category,
            title=req.title,
            response_code=200,
            response_msg="知识库创建成功",
            process_status="completed",
            processing_time=time.time() - start_time,
        )
    except Exception as e:
        logger.error(f"知识库创建失败: {e}", exc_info=True)

    return KnowledgeResponse(
        request_id=str(uuid.uuid4()),
        knowledge_id=0,
        category="",
        title="",
        response_code=500,
        response_msg="知识库创建失败",
        process_status="failed",
        processing_time=time.time() - start_time,
    )


@router.get("/knowledge_base", response_model=KnowledgeResponse)
def get_knowledge_base(knowledge_id: int, token: str) -> KnowledgeResponse:
    start_time = time.time()
    try:
        with Session() as session:
            record = session.query(KnowledgeDatabase).filter(
                KnowledgeDatabase.knowledge_id == knowledge_id
            ).first()
        if record is not None:
            return KnowledgeResponse(
                request_id=str(uuid.uuid4()),
                knowledge_id=knowledge_id,
                title=str(record.title),
                category=str(record.category),
                response_code=200,
                response_msg="知识库查询成功",
                process_status="completed",
                processing_time=time.time() - start_time,
            )
    except Exception as e:
        logger.error(f"知识库查询失败 knowledge_id={knowledge_id}: {e}", exc_info=True)

    return KnowledgeResponse(
        request_id=str(uuid.uuid4()),
        knowledge_id=knowledge_id,
        category="",
        title="",
        response_code=404,
        response_msg="知识库不存在",
        process_status="completed",
        processing_time=time.time() - start_time,
    )


@router.delete("/knowledge_base", response_model=KnowledgeResponse)
def delete_knowledge_base(knowledge_id: int, token: str) -> KnowledgeResponse:
    start_time = time.time()
    try:
        with Session() as session:
            record = session.query(KnowledgeDatabase).filter(
                KnowledgeDatabase.knowledge_id == knowledge_id
            ).first()
            if record is not None:
                session.delete(record)
                session.commit()
                logger.info(f"知识库删除成功 knowledge_id={knowledge_id}")
                return KnowledgeResponse(
                    request_id=str(uuid.uuid4()),
                    knowledge_id=knowledge_id,
                    category=str(record.category),
                    title=str(record.title),
                    response_code=200,
                    response_msg="知识库删除成功",
                    process_status="completed",
                    processing_time=time.time() - start_time,
                )
    except Exception as e:
        logger.error(f"知识库删除失败 knowledge_id={knowledge_id}: {e}", exc_info=True)

    return KnowledgeResponse(
        request_id=str(uuid.uuid4()),
        knowledge_id=knowledge_id,
        category="",
        title="",
        response_code=404,
        response_msg="知识库不存在",
        process_status="completed",
        processing_time=time.time() - start_time,
    )
