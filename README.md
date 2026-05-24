# Advanced RAG System

企业知识库检索增强生成（RAG）问答系统。支持 PDF / Word 文档入库，多知识库管理，双路召回 + RRF 融合 + 查询改写，基于 Qwen 大模型生成答案。

## 系统架构

```
用户提问
   ↓
查询改写（LLM 将口语化问题转为检索关键词）
   ↓
双路召回
  ├─ BM25 全文检索（IK 中文分词）
  └─ KNN 语义检索（bge-small-zh-v1.5 + HNSW）
   ↓
RRF 融合（Reciprocal Rank Fusion，k=60）
   ↓
[可选] Cross-Encoder 重排序（bge-reranker-base）
   ↓
LLM 生成（Qwen，注入检索上下文）
   ↓
返回答案
```

**技术栈**：FastAPI · Elasticsearch 8 · SQLAlchemy · SentenceTransformers · Qwen（阿里云百炼）

## 快速开始（本地）

### 前提条件
- Docker & Docker Compose
- Python 3.11+
- 已下载模型文件（见下方）

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 DASHSCOPE_API_KEY
```

### 2. 放入模型文件

将已下载的模型文件分别放入：

```
models/BAAI/bge-small-zh-v1.5/    ← bge-small-zh-v1.5 模型文件
models/BAAI/bge-reranker-base/    ← bge-reranker-base 模型文件
```

模型下载地址：
- [bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- [bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)

### 3. 启动 Elasticsearch

```bash
docker compose up -d elasticsearch
```

首次启动会自动安装 IK 中文分词插件（约需 60 秒），等待健康检查通过：

```bash
curl http://localhost:9200/_cluster/health
# 返回 "status":"green" 或 "yellow" 即可
```

### 4. 安装依赖

```bash
pip install -r requirements.txt
```

### 5. 启动服务

```bash
python main.py
```

- API 文档：http://localhost:6006/docs
- 健康检查：http://localhost:6006/health

### 6. 验证完整链路

通过 Swagger UI（`/docs`）依次操作：

1. `POST /v1/knowledge_base` — 创建知识库
2. `POST /v1/document` — 上传 PDF/Word 文档
3. `POST /chat` — 提问，观察 RAG 检索效果

## 运行测试

```bash
pytest tests/test_db.py -v        # 数据库测试（无外部依赖）
pytest tests/test_es.py -v        # ES 测试（需要 ES 运行）
pytest tests/test_rag.py -v       # RAG 测试（需要模型文件 + 百炼 API）
pytest tests/test_api.py -v       # 完整接口测试
pytest tests/ -v                  # 全部测试
```

## 目录结构

```
├── app/
│   ├── api/          # HTTP 路由（knowledge / document / chat / embed）
│   ├── core/         # 业务逻辑（RAG 流程、文档提取、模型推理）
│   ├── db/           # 数据访问（SQLAlchemy ORM、Elasticsearch 初始化）
│   ├── schemas/      # Pydantic 请求/响应模型
│   └── config.py     # 统一配置加载（config.yaml + .env）
├── models/BAAI/      # 本地模型（git 仅追踪目录结构）
├── tests/            # 测试套件
├── upload_files/     # 上传文档存储目录
├── config.yaml       # 非敏感配置
├── .env.example      # 环境变量模板
├── docker-compose.yml
└── main.py           # 应用入口
```

## 配置说明

`config.yaml` 包含所有非敏感配置，敏感信息通过 `.env` 提供：

| 环境变量 | 说明 |
|----------|------|
| `DASHSCOPE_API_KEY` | 阿里云百炼 API Key（必填，在百炼控制台获取） |
| `ES_HOST` | Elasticsearch 主机（默认 localhost） |

如需启用 GPU 加速，将 `config.yaml` 中 `device: "cpu"` 改为 `device: "cuda"`。
