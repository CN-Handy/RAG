import logging
import logging.handlers
import datetime

import uvicorn
from fastapi import FastAPI

from app.config import config
from app.core.rag import EMBEDDING_MODEL_PARAMS
from app.db.vector_store import es
from app.api.knowledge import router as knowledge_router
from app.api.document import router as document_router
from app.api.chat import router as chat_router
from app.api.embed import router as embed_router


def setup_logging() -> None:
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.setLevel(logging.INFO)

    file_handler = logging.handlers.RotatingFileHandler(
        "rag_service.log", maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)


setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Advanced RAG API",
    version="1.0.0",
    description="企业知识库检索增强生成（RAG）问答系统",
)

app.include_router(knowledge_router, prefix="/v1", tags=["知识库管理"])
app.include_router(document_router, prefix="/v1", tags=["文档管理"])
app.include_router(chat_router, tags=["对话"])
app.include_router(embed_router, prefix="/v1", tags=["模型工具"])


@app.get("/health", tags=["运维"])
def health_check() -> dict:
    components: dict = {}
    ok = True

    try:
        es_ok = es.ping()
        components["elasticsearch"] = {"status": "ok" if es_ok else "error"}
        if not es_ok:
            ok = False
    except Exception as e:
        components["elasticsearch"] = {"status": "error", "detail": str(e)}
        ok = False

    components["embedding_model"] = {
        "status": "ok" if "embedding_model" in EMBEDDING_MODEL_PARAMS else "not_loaded",
        "model": config["rag"]["embedding_model"],
    }

    return {
        "status": "ok" if ok else "degraded",
        "timestamp": str(datetime.datetime.now()),
        "components": components,
    }


if __name__ == "__main__":
    logger.info("启动 Advanced RAG 服务...")
    uvicorn.run(app, host="0.0.0.0", port=config["rag"]["port"], workers=1)
