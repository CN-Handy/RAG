import uuid
import time
import logging

import numpy as np
from fastapi import APIRouter

from app.core.rag import RAG
from app.schemas.api import EmbeddingRequest, EmbeddingResponse, RerankRequest, RerankResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/embedding", response_model=EmbeddingResponse)
async def semantic_embedding(req: EmbeddingRequest) -> EmbeddingResponse:
    start_time = time.time()
    text = req.text if isinstance(req.text, list) else [req.text]
    vector: np.ndarray = RAG().get_embedding(text)
    return EmbeddingResponse(
        request_id=str(uuid.uuid4()),
        vector=vector.astype(float).tolist(),
        response_code=200,
        response_msg="ok",
        process_status="completed",
        processing_time=time.time() - start_time,
    )


@router.post("/rerank", response_model=RerankResponse)
async def semantic_rerank(req: RerankRequest) -> RerankResponse:
    start_time = time.time()
    vector: np.ndarray = RAG().get_rank(req.text_pair)
    return RerankResponse(
        request_id=str(uuid.uuid4()),
        vector=vector.astype(float).tolist(),
        response_code=200,
        response_msg="ok",
        process_status="completed",
        processing_time=time.time() - start_time,
    )
