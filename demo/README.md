# Advanced RAG — Streamlit Demo

基于 Streamlit 的可视化演示界面，覆盖知识库管理、文档上传与解析状态跟踪、RAG 多轮问答全流程。

## 快速启动

### 1. 确保后端服务运行中

```bash
# 在项目根目录执行
python main.py
# 服务启动在 http://localhost:6006
```

### 2. 安装 Demo 依赖

```bash
pip install streamlit requests
# 或使用 demo/requirements.txt
pip install -r demo/requirements.txt
```

### 3. 启动 Demo

```bash
# 在项目根目录执行
streamlit run demo/app.py
```

浏览器会自动打开 http://localhost:8501

## 功能说明

| 标签页 | 功能 |
|--------|------|
| 💬 RAG 问答 | 多轮对话；查询改写 + BM25/KNN 双路召回 + RRF 融合 + Cross-Encoder 重排序 |
| 📄 文档管理 | 上传 PDF/Word，实时查看 `parse_status`（待处理 → 解析中 → 已就绪），删除文档 |
| 🗂️ 知识库管理 | 创建/删除知识库，在侧边栏快速切换当前知识库 |

## 使用流程

1. 侧边栏确认服务状态为 🟢
2. 在「知识库管理」创建一个知识库
3. 在「文档管理」上传 PDF 或 Word 文档，等待 `parse_status` 变为 🟢 已就绪
4. 在「RAG 问答」提问，观察检索增强效果
