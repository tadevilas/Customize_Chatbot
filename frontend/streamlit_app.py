import os
import streamlit as st
import requests
import json
import uuid

# Allow override via environment variable (used in Docker / production)
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Agentic AI Platform",
    page_icon="🤖",
    layout="wide"
)

# Custom header
st.title("🤖 Production-Ready Agentic AI Platform")
st.caption("Powered by Groq LLMs, LangGraph, SQLite & Tavily MCP Servers")

# Session State Initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "load_session_id" not in st.session_state:
    st.session_state.load_session_id = None

if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}   # {msg_index: "helpful" | "not_helpful"}

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
    
    if st.button("🔄 New Chat Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.subheader("🕓 Recent Sessions")
    try:
        sessions_resp = requests.get(f"{BACKEND_URL}/sessions?limit=3", timeout=3)
        if sessions_resp.status_code == 200:
            recent_sessions = sessions_resp.json()
            if recent_sessions:
                for sess in recent_sessions:
                    sid = sess["session_id"]
                    preview = sess["preview"]
                    created = sess["created_at"][:16].replace("T", " ")
                    label = f"🗨️ {preview}"
                    is_active = sid == st.session_state.session_id
                    btn_type = "primary" if is_active else "secondary"
                    if st.button(label, key=f"sess_{sid}", type=btn_type, help=f"Started: {created} | ID: {sid[:8]}…", use_container_width=True):
                        if not is_active:
                            # Load messages from this session
                            hist_resp = requests.get(f"{BACKEND_URL}/sessions/{sid}/messages", timeout=5)
                            if hist_resp.status_code == 200:
                                loaded_msgs = [
                                    {"role": m["role"], "content": m["message"], "tools_used": []}
                                    for m in hist_resp.json()
                                ]
                                st.session_state.session_id = sid
                                st.session_state.messages = loaded_msgs
                                st.rerun()
            else:
                st.caption("No previous sessions found.")
        else:
            st.caption("Could not load recent sessions.")
    except Exception:
        st.caption("Backend offline — sessions unavailable.")

    st.markdown("---")
    st.subheader("📄 Upload Documents (RAG)")
    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX or TXT",
        type=["pdf", "docx", "txt"],
        help="Uploaded documents are indexed into ChromaDB for RAG retrieval."
    )
    if uploaded_file:
        if st.button("📥 Ingest Document", use_container_width=True):
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                        timeout=120
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(
                            f"✅ **{data['filename']}** ingested!\n\n"
                            f"- Docs extracted: {data['docs_extracted']}\n"
                            f"- Chunks stored: {data['chunks_stored']}\n"
                            f"- Content types: {data['content_types']}"
                        )
                    else:
                        st.error(f"Upload failed: {resp.json().get('detail', resp.text)}")
                except Exception as e:
                    st.error(f"Upload error: {e}")

    # Show already-uploaded documents
    try:
        docs_resp = requests.get(f"{BACKEND_URL}/upload/documents", timeout=3)
        if docs_resp.status_code == 200:
            doc_data = docs_resp.json()
            if doc_data.get("documents"):
                st.caption(f"**Indexed documents ({doc_data['count']}):**")
                for doc in doc_data["documents"]:
                    col1, col2 = st.columns([4, 1])
                    col1.caption(f"📎 {doc['filename']} ({doc['size_kb']} KB)")
                    if col2.button("🗑️", key=f"del_{doc['filename']}", help="Delete file"):
                        requests.delete(f"{BACKEND_URL}/upload/documents/{doc['filename']}", timeout=5)
                        st.rerun()
    except Exception:
        pass

    st.markdown("---")
    st.subheader("🛠️ System Status")
    try:
        health_resp = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            st.success("Backend: Online ✅")
            st.write(f"**Primary Model:** `{health_data.get('primary_model')}`")
            st.write("**MCP SQLite:** Active 🟢")
            st.write("**MCP Tavily Search:** Active 🌐")
            st.write("**RAG (ChromaDB + BGE):** Active 📚")

            # LangSmith tracing status
            st.markdown("---")
            if health_data.get("langsmith_tracing"):
                project = health_data.get("langsmith_project", "")
                st.success("**LangSmith:** Tracing ON 🔭")
                st.caption(f"Project: `{project}`")
                langsmith_url = f"https://smith.langchain.com/projects/{project}"
                st.markdown(f"[Open LangSmith Dashboard ↗]({langsmith_url})")
            else:
                st.warning("**LangSmith:** Tracing OFF ⚠️")
                st.caption("Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` in `.env` to enable.")
        else:
            st.warning("Backend responding with non-200 code.")
    except Exception:
        st.error("Backend: Offline ❌ (Run `uv run uvicorn backend.main:app`)")

# Display Message History
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "tools_used" in msg and msg["tools_used"]:
            with st.expander(f"🔍 MCP Tools Executed ({len(msg['tools_used'])})"):
                st.json(msg["tools_used"])

        # Feedback buttons — only on assistant messages
        if msg["role"] == "assistant":
            already_rated = st.session_state.feedback_given.get(idx)
            if already_rated:
                st.caption(f"{'👍' if already_rated == 'helpful' else '👎'} Feedback recorded — thank you!")
            else:
                # Find the user question that preceded this assistant message
                question = ""
                if idx > 0 and st.session_state.messages[idx - 1]["role"] == "user":
                    question = st.session_state.messages[idx - 1]["content"]

                col1, col2, col3 = st.columns([1, 1, 8])
                if col1.button("👍", key=f"helpful_{idx}", help="This was helpful"):
                    try:
                        requests.post(f"{BACKEND_URL}/feedback", json={
                            "question": question,
                            "response": msg["content"],
                            "rating": "helpful",
                            "comments": ""
                        }, timeout=5)
                        st.session_state.feedback_given[idx] = "helpful"
                        st.rerun()
                    except Exception:
                        st.toast("Could not save feedback — backend offline.", icon="⚠️")

                if col2.button("👎", key=f"not_helpful_{idx}", help="This was not helpful"):
                    try:
                        requests.post(f"{BACKEND_URL}/feedback", json={
                            "question": question,
                            "response": msg["content"],
                            "rating": "not_helpful",
                            "comments": ""
                        }, timeout=5)
                        st.session_state.feedback_given[idx] = "not_helpful"
                        st.rerun()
                    except Exception:
                        st.toast("Could not save feedback — backend offline.", icon="⚠️")

# Chat Input with Real-Time Streaming
if prompt := st.chat_input("Ask a question (e.g. 'Show total sales by product' or 'What is the latest AI news today?')"):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI backend streaming endpoint
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_container = st.status("Thinking & querying MCP tools...", expanded=True)
        
        full_response = ""
        tools_executed = []
        latency = 0.0

        try:
            with requests.post(
                f"{BACKEND_URL}/chat/stream",
                json={
                    "session_id": st.session_state.session_id,
                    "user_id": "streamlit_user",
                    "message": prompt
                },
                stream=True,
                timeout=120
            ) as response:
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if not line:
                            continue
                        line_text = line.decode("utf-8")
                        if line_text.startswith("data: "):
                            event_data = json.loads(line_text[6:])
                            event_type = event_data.get("type")

                            if event_type == "token":
                                full_response += event_data.get("content", "")
                                message_placeholder.markdown(full_response + "▌")
                            elif event_type == "agent_start":
                                status_container.write("🧠 Supervisor analysing query...")
                            elif event_type == "route":
                                agents = event_data.get("agents", [])
                                agent_labels = {"sql": "🗄️ SQL Agent", "search": "🌐 Research Agent", "rag": "📚 RAG Agent", "direct": "💬 Direct Answer"}
                                routed_to = " + ".join(agent_labels.get(a, a) for a in agents)
                                status_container.write(f"🔀 Routed to: **{routed_to}**")
                            elif event_type == "tool_start":
                                tool_info = event_data.get("tool", {})
                                tools_executed.append(tool_info)
                                status_container.write(f"🛠️ Executing `{tool_info.get('tool')}` with arguments: `{tool_info.get('args')}`")
                            elif event_type == "tool_end":
                                tool_name = event_data.get("tool", "")
                                status_container.write(f"✅ Completed `{tool_name}`")
                            elif event_type == "final_stats":
                                latency = event_data.get("latency", 0.0)

                    status_container.update(label=f"Done in {latency}s! ({len(tools_executed)} tools used)", state="complete", expanded=False)
                    message_placeholder.markdown(full_response)

                    if tools_executed:
                        with st.expander(f"🔍 MCP Tools Executed ({len(tools_executed)}) — Latency: {latency}s"):
                            st.json(tools_executed)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "tools_used": tools_executed,
                        "question": prompt
                    })
                else:
                    error_text = f"API Error ({response.status_code}): {response.text}"
                    status_container.update(label="Error occurred", state="error")
                    st.error(error_text)
                    st.session_state.messages.append({"role": "assistant", "content": error_text})

        except Exception as e:
            err_msg = f"Failed to connect to backend: {str(e)}"
            status_container.update(label="Connection Error", state="error")
            st.error(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})
