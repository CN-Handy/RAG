import uuid
import time
import logging

from fastapi import APIRouter

from app.core.rag import RAG
from app.schemas.api import RAGRequest, RAGResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=RAGResponse)
def chat(req: RAGRequest) -> RAGResponse:
    start_time = time.time()
    try:
        messages = RAG().chat_with_rag(req.knowledge_id, req.message)
        return RAGResponse(
            request_id=str(uuid.uuid4()),
            message=messages,
            response_code=200,
            response_msg="ok",
            process_status="completed",
            processing_time=time.time() - start_time,
        )
    except Exception as e:
        logger.error(f"对话失败: {e}", exc_info=True)
        return RAGResponse(
            request_id=str(uuid.uuid4()),
            message=req.message,
            response_code=500,
            response_msg="对话失败，请检查服务状态",
            process_status="failed",
            processing_time=time.time() - start_time,
        )
