import logging
import traceback
from elasticsearch import Elasticsearch

from app.config import config

logger = logging.getLogger(__name__)

_es_cfg = config["elasticsearch"]
_auth = (_es_cfg["username"], _es_cfg["password"]) if _es_cfg["username"] else None

es = Elasticsearch(
    f"{_es_cfg['scheme']}://{_es_cfg['host']}:{_es_cfg['port']}",

    basic_auth=_auth if _auth else None,

    # 关闭 SSL 校验（http 必须关）
    verify_certs=False,
    ssl_show_warn=False
)

_embedding_dims = config["models"]["embedding_model"][config["rag"]["embedding_model"]]["dims"]


def init_es() -> bool:
    if not es.ping():
        es_address = f"{_es_cfg['scheme']}://{_es_cfg['host']}:{_es_cfg['port']}"
        logger.error(f"无法连接到 Elasticsearch，请确认服务已启动")
        logger.error(f"Elasticsearch 地址: {es_address}")
        return False

    _document_meta_mapping = {
        "mappings": {
            "properties": {
                "document_id":   {"type": "integer"},
                "knowledge_id":  {"type": "integer"},
                "document_name": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_max_word"},
                "file_path":     {"type": "keyword"},
                "abstract":      {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_max_word"},
            }
        }
    }
    try:
        if not es.indices.exists(index="document_meta"):
            es.indices.create(index="document_meta", body=_document_meta_mapping)
            logger.info("索引 document_meta 创建成功")
    except Exception:
        logger.error("创建索引 document_meta 失败\n" + traceback.format_exc())
        return False

    _chunk_info_mapping = {
        "mappings": {
            "properties": {
                "chunk_content": {
                    "type": "text",
                    "analyzer": "ik_max_word",
                    "search_analyzer": "ik_max_word",
                },
                "embedding_vector": {
                    "type": "dense_vector",
                    "element_type": "float",
                    "dims": _embedding_dims,
                    "index": True,
                    "index_options": {"type": "int8_hnsw"},
                },
            }
        }
    }
    try:
        if not es.indices.exists(index="chunk_info"):
            es.indices.create(index="chunk_info", body=_chunk_info_mapping)
            logger.info("索引 chunk_info 创建成功")
    except Exception:
        logger.error("创建索引 chunk_info 失败\n" + traceback.format_exc())
        return False

    logger.info("Elasticsearch 初始化完成")
    return True


init_es()
