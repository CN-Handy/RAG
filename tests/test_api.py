from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_knowledge_base():
    # 创建
    resp = client.post("/v1/knowledge_base", json={"category": "测试类别", "title": "测试标题"})
    assert resp.status_code == 200
    assert resp.json()["response_code"] == 200

    knowledge_id = resp.json()["knowledge_id"]

    # 查询
    resp = client.get(f"/v1/knowledge_base?knowledge_id={knowledge_id}&token=test")
    assert resp.status_code == 200
    assert resp.json()["title"] == "测试标题"

    # 删除
    resp = client.delete(f"/v1/knowledge_base?knowledge_id={knowledge_id}&token=test")
    assert resp.status_code == 200
    assert resp.json()["response_msg"] == "知识库删除成功"

    # 再次查询应不存在
    resp = client.get(f"/v1/knowledge_base?knowledge_id={knowledge_id}&token=test")
    assert resp.json()["response_code"] == 404


def test_document_not_found():
    resp = client.get("/v1/document?document_id=99999&token=test")
    assert resp.status_code == 200
    assert resp.json()["response_msg"] == "文档不存在"

    resp = client.delete("/v1/document?document_id=99999&token=test")
    assert resp.status_code == 200
    assert resp.json()["response_msg"] == "文档不存在"


def test_health():
    resp = client.get("/health")
    assert resp.status_code in (200, 503)
    assert "status" in resp.json()
    assert "components" in resp.json()
