import numpy as np
from app.db.vector_store import es


def test_connect_es():
    assert es.ping(), "Elasticsearch 无法连接，请确认服务已启动"


def test_init_es():
    assert es.indices.exists(index="document_meta"), "索引 document_meta 不存在"
    assert es.indices.exists(index="chunk_info"), "索引 chunk_info 不存在"


def test_insert_document_meta():
    doc = {
        "file_path": "test_file.txt",
        "file_name": "测试文件",
        "abstract": "这是一个测试摘要。",
        "full_content": "这是测试文件的完整内容。",
    }
    resp = es.index(index="document_meta", document=doc)
    assert resp["result"] == "created"
    doc_id = resp["_id"]

    assert es.exists(index="document_meta", id=doc_id)
    es.delete(index="document_meta", id=doc_id)


def test_query_document_meta():
    doc = {
        "file_path": "query_test.txt",
        "file_name": "查询测试文件",
        "abstract": "用于查询测试的摘要。",
        "full_content": "查询测试的完整内容。",
    }
    resp = es.index(index="document_meta", document=doc, refresh=True)
    doc_id = resp["_id"]

    search = es.search(index="document_meta", query={"match": {"file_name": "查询测试文件"}})
    assert search["hits"]["total"]["value"] > 0

    es.delete(index="document_meta", id=doc_id)


def test_insert_chunk_info():
    chunk = {
        "chunk_id": 0,
        "knowledge_id": 1,
        "document_id": 1,
        "page_number": 1,
        "chunk_content": "这是测试片段的内容。",
        "chunk_images": [],
        "chunk_tables": [],
        "embedding_vector": [0.1] * 512,
    }
    resp = es.index(index="chunk_info", document=chunk)
    assert resp["result"] == "created"
    doc_id = resp["_id"]

    assert es.exists(index="chunk_info", id=doc_id)
    es.delete(index="chunk_info", id=doc_id)


def test_query_chunk_info():
    chunk = {
        "chunk_id": 0,
        "knowledge_id": 1,
        "document_id": 1,
        "page_number": 1,
        "chunk_content": "企业知识库文档检索测试",
        "chunk_images": [],
        "chunk_tables": [],
        "embedding_vector": np.random.rand(512).tolist(),
    }
    resp = es.index(index="chunk_info", document=chunk, refresh=True)
    doc_id = resp["_id"]

    # BM25 关键词检索
    search = es.search(index="chunk_info", body={"query": {"match": {"chunk_content": "知识库文档"}}})
    assert search["hits"]["total"]["value"] > 0

    # KNN 向量检索
    knn_resp = es.search(
        index="chunk_info",
        knn={"field": "embedding_vector", "query_vector": [0.1] * 512, "k": 5, "num_candidates": 10},
    )
    assert knn_resp["hits"]["total"]["value"] > 0

    es.delete(index="chunk_info", id=doc_id)
