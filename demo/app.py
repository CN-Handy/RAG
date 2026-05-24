"""
Advanced RAG System — Streamlit Demo
Run from project root: streamlit run demo/app.py
"""
import time

import requests
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Advanced RAG Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal custom CSS ────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] { min-width: 280px; max-width: 320px; }
        div[data-testid="stChatMessage"] { padding: 0.6rem 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

_TOKEN = "demo"


# ── API helpers ───────────────────────────────────────────────────────────────

def _base() -> str:
    return st.session_state.get("api_base", "http://localhost:6006").rstrip("/")


def api_get(path: str, **params):
    try:
        r = requests.get(f"{_base()}{path}", params={"token": _TOKEN, **params}, timeout=10)
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "无法连接服务，请先运行 `python main.py` 启动后端"
    except Exception as exc:
        return None, str(exc)


def api_post_json(path: str, payload: dict):
    try:
        r = requests.post(f"{_base()}{path}", json=payload, timeout=120)
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "无法连接服务，请先运行 `python main.py` 启动后端"
    except Exception as exc:
        return None, str(exc)


def api_post_form(path: str, data: dict, files=None):
    try:
        r = requests.post(f"{_base()}{path}", data=data, files=files, timeout=120)
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "无法连接服务，请先运行 `python main.py` 启动后端"
    except Exception as exc:
        return None, str(exc)


def api_delete(path: str, **params):
    try:
        r = requests.delete(f"{_base()}{path}", params={"token": _TOKEN, **params}, timeout=10)
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "无法连接服务，请先运行 `python main.py` 启动后端"
    except Exception as exc:
        return None, str(exc)


def _parse_status_label(status: str) -> str:
    return {
        "completed": "🟢 已就绪",
        "processing": "🟡 解析中…",
        "pending":    "⚪ 待处理",
        "failed":     "🔴 解析失败",
    }.get(status, f"❓ {status}")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Advanced RAG")
    st.text_input("API 服务地址", "http://localhost:6006", key="api_base")

    # Health check — blocks the page if service is down
    health, health_err = api_get("/health")
    if health_err:
        st.error(f"🔴 **服务离线**\n\n{health_err}")
        st.code("python main.py", language="bash")
        st.stop()

    overall = health.get("status", "unknown")
    es_status = health.get("components", {}).get("elasticsearch", {}).get("status", "unknown")
    model_status = health.get("components", {}).get("embedding_model", {}).get("status", "unknown")

    if overall == "ok":
        st.success("🟢 服务运行正常")
    else:
        st.warning("🟡 服务部分降级")

    with st.expander("组件状态"):
        st.write(f"**Elasticsearch**：{'✅ 正常' if es_status == 'ok' else '❌ 异常'}")
        st.write(f"**Embedding 模型**：{'✅ 已加载' if model_status == 'ok' else '⚠️ 未加载（影响语义检索）'}")

    st.divider()

    # Knowledge base selector
    st.markdown("**📚 当前知识库**")
    kb_resp, _ = api_get("/v1/knowledge_bases")
    kb_list = (kb_resp or {}).get("knowledge_bases", [])

    if not kb_list:
        st.info("暂无知识库\n\n→ 前往「知识库管理」标签页创建")
        st.session_state.setdefault("cur_kb_id", None)
        st.session_state.setdefault("cur_kb_title", "")
    else:
        kb_opts = {f"[{kb['knowledge_id']}]  {kb['title']}": kb for kb in kb_list}

        # Restore previously selected entry if it still exists
        default_idx = 0
        if st.session_state.get("cur_kb_id"):
            for i, kb in enumerate(kb_list):
                if kb["knowledge_id"] == st.session_state["cur_kb_id"]:
                    default_idx = i
                    break

        chosen = st.selectbox(
            "选择知识库",
            list(kb_opts),
            index=default_idx,
            label_visibility="collapsed",
        )
        st.session_state["cur_kb_id"] = kb_opts[chosen]["knowledge_id"]
        st.session_state["cur_kb_title"] = kb_opts[chosen]["title"]


# ── Shared state (read after sidebar so cur_kb_id is always set) ──────────────
cur_kb_id: int | None = st.session_state.get("cur_kb_id")
cur_kb_title: str = st.session_state.get("cur_kb_title", "")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_docs, tab_kb = st.tabs(
    ["💬  RAG 问答", "📄  文档管理", "🗂️  知识库管理"]
)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — RAG Chat
# ═════════════════════════════════════════════════════════════════════════════
with tab_chat:
    if not cur_kb_id:
        st.info("👈 请先在左侧侧边栏选择或创建知识库，然后开始对话")
    else:
        # Header
        hc1, hc2 = st.columns([7, 1])
        with hc1:
            st.subheader(f"与「{cur_kb_title}」对话")
        with hc2:
            st.write("")  # vertical align
            if st.button("🗑️ 清空", help="清空本次对话"):
                st.session_state[f"hist_{cur_kb_id}"] = []
                st.rerun()

        hist_key = f"hist_{cur_kb_id}"
        if hist_key not in st.session_state:
            st.session_state[hist_key] = []
        hist: list = st.session_state[hist_key]

        # Render existing messages
        for msg in hist:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # New message input
        user_query = st.chat_input("输入问题，Enter 发送…", key=f"chat_input_{cur_kb_id}")
        if user_query:
            with st.chat_message("user"):
                st.markdown(user_query)

            pending_hist = hist + [{"role": "user", "content": user_query}]

            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown("⏳ 检索文档并生成回答中…")

                resp, err = api_post_json("/chat", {
                    "knowledge_id": cur_kb_id,
                    "message": pending_hist,
                })

                if err:
                    placeholder.error(f"请求失败：{err}")
                elif resp and resp.get("response_code") == 200:
                    reply = resp["message"][-1]["content"]
                    placeholder.markdown(reply)
                    st.session_state[hist_key] = resp["message"]
                    st.caption(f"⚡ 耗时 {resp.get('processing_time', 0):.2f}s")
                else:
                    msg_text = (resp or {}).get("response_msg", "服务异常，请检查后端日志")
                    placeholder.error(f"❌ {msg_text}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Document Management
# ═════════════════════════════════════════════════════════════════════════════
with tab_docs:
    if not cur_kb_id:
        st.info("👈 请先在左侧侧边栏选择知识库")
    else:
        st.subheader(f"📄 「{cur_kb_title}」— 文档管理")

        # ── Upload form ───────────────────────────────────────────────────────
        with st.expander("➕ 上传新文档", expanded=False):
            uc1, uc2 = st.columns(2)
            with uc1:
                up_title = st.text_input("文档标题 *", placeholder="如：产品手册 v2.0", key="up_title")
            with uc2:
                up_cat = st.text_input("分类", value="通用", key="up_cat")
            up_file = st.file_uploader(
                "选择文件（支持 PDF / Word .docx）",
                type=["pdf", "docx"],
                key="up_file",
            )
            if st.button("上传", type="primary", disabled=not (up_title and up_file)):
                with st.spinner("上传并提交解析任务…"):
                    result, err = api_post_form(
                        "/v1/document",
                        data={
                            "knowledge_id": str(cur_kb_id),
                            "title": up_title,
                            "category": up_cat,
                        },
                        files={"file": (up_file.name, up_file.getvalue(), up_file.type)},
                    )
                if err:
                    st.error(f"上传失败：{err}")
                elif result and result.get("response_code") == 200:
                    st.success(
                        f"✅ 上传成功！document_id={result['document_id']}，后台解析中…"
                    )
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.error((result or {}).get("response_msg", "上传失败，请检查后端日志"))

        # ── Document list ─────────────────────────────────────────────────────
        rc1, rc2 = st.columns([1, 7])
        with rc1:
            if st.button("🔄 刷新状态"):
                st.rerun()

        doc_resp, doc_err = api_get("/v1/documents", knowledge_id=cur_kb_id)

        if doc_err:
            st.error(f"加载失败：{doc_err}")
        elif not doc_resp or not doc_resp.get("documents"):
            st.info("该知识库暂无文档，请先上传文件")
        else:
            docs: list = doc_resp["documents"]
            has_pending = any(d["parse_status"] in ("processing", "pending") for d in docs)

            if has_pending:
                st.info("⏳ 有文档正在解析，点击「🔄 刷新状态」查看最新进度")

            st.markdown(f"共 **{len(docs)}** 份文档")
            for doc in docs:
                with st.container(border=True):
                    dc1, dc2 = st.columns([7, 1])
                    with dc1:
                        status_label = _parse_status_label(doc["parse_status"])
                        st.markdown(f"**{doc['title']}** &nbsp;&nbsp; {status_label}")
                        ft = doc["file_type"]
                        short_type = ft.split("/")[-1] if "/" in ft else ft
                        st.caption(
                            f"document_id: {doc['document_id']}  ·  "
                            f"分类: {doc['category']}  ·  "
                            f"类型: {short_type}"
                        )
                    with dc2:
                        if st.button("删除", key=f"ddoc_{doc['document_id']}"):
                            r, e = api_delete("/v1/document", document_id=doc["document_id"])
                            if e:
                                st.error(e)
                            elif r and r.get("response_code") == 200:
                                st.toast("已删除")
                                time.sleep(0.4)
                                st.rerun()
                            else:
                                st.error((r or {}).get("response_msg", "删除失败"))


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Knowledge Base Management
# ═════════════════════════════════════════════════════════════════════════════
with tab_kb:
    st.subheader("🗂️ 知识库管理")

    # ── Create form ───────────────────────────────────────────────────────────
    with st.expander("➕ 创建新知识库", expanded=not kb_list):
        kc1, kc2 = st.columns(2)
        with kc1:
            new_title = st.text_input("名称 *", placeholder="如：技术文档库", key="new_kb_title")
        with kc2:
            new_cat = st.text_input("分类 *", placeholder="如：研发", key="new_kb_cat")
        if st.button("创建", type="primary", disabled=not (new_title and new_cat)):
            r, e = api_post_json("/v1/knowledge_base", {
                "title": new_title,
                "category": new_cat,
            })
            if e:
                st.error(f"创建失败：{e}")
            elif r and r.get("response_code") == 200:
                st.success(f"✅ 知识库「{new_title}」已创建（ID: {r['knowledge_id']}）")
                st.session_state["cur_kb_id"] = r["knowledge_id"]
                st.session_state["cur_kb_title"] = new_title
                time.sleep(0.5)
                st.rerun()
            else:
                st.error((r or {}).get("response_msg", "创建失败"))

    # ── Knowledge base list ───────────────────────────────────────────────────
    if not kb_list:
        st.info("暂无知识库，请先创建")
    else:
        st.markdown(f"共 **{len(kb_list)}** 个知识库")
        for kb in kb_list:
            with st.container(border=True):
                kl1, kl2 = st.columns([7, 1])
                with kl1:
                    is_cur = kb["knowledge_id"] == cur_kb_id
                    name_text = f"**{kb['title']}**" + ("  ← 当前选中" if is_cur else "")
                    st.markdown(name_text)
                    st.caption(f"ID: {kb['knowledge_id']}  ·  分类: {kb['category']}")
                with kl2:
                    if st.button("删除", key=f"dkb_{kb['knowledge_id']}"):
                        r, e = api_delete("/v1/knowledge_base", knowledge_id=kb["knowledge_id"])
                        if e:
                            st.error(e)
                        elif r and r.get("response_code") == 200:
                            st.toast(f"已删除「{kb['title']}」")
                            if st.session_state.get("cur_kb_id") == kb["knowledge_id"]:
                                st.session_state["cur_kb_id"] = None
                                st.session_state["cur_kb_title"] = ""
                            time.sleep(0.4)
                            st.rerun()
                        else:
                            st.error((r or {}).get("response_msg", "删除失败"))
