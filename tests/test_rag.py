import numpy as np
import pytest
from app.config import config
from app.core.rag import RAG


def test_embedding_model():
    if not config["rag"]["use_embedding"]:
        pytest.skip("use_embedding 未启用")
    embedding = RAG().get_embedding("企业知识库文档检索测试")
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[-1] == config["models"]["embedding_model"][
        config["rag"]["embedding_model"]
    ]["dims"]


def test_rerank_model():
    if not config["rag"]["use_rerank"]:
        pytest.skip("use_rerank 未启用")
    pairs = [
        ["我今天很开心", "我今天很开心"],
        ["我今天很开心", "今天天气真差"],
    ]
    scores = RAG().get_rank(pairs)
    assert isinstance(scores, np.ndarray)
    assert scores[0] > scores[1], "相同文本的相关性应高于不相关文本"


def test_query_rewrite():
    rewritten = RAG().query_rewrite("这份合同的签署流程是什么")
    assert isinstance(rewritten, str)
    assert len(rewritten) > 0


def test_llm():
    messages = [
        {"role": "system", "content": "你是一个专业的知识库问答助手"},
        {"role": "user", "content": "请用一句话介绍什么是 RAG（检索增强生成）"},
    ]
    result = RAG().chat(messages, top_p=0.9, temperature=0.1)
    assert result is not None
    assert result.content is not None and len(result.content) > 0
