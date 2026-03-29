import time
import os

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")


st.set_page_config(
    page_title="StackAI RAG",
    layout="wide",
    initial_sidebar_state="expanded",
)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "ingested_files" not in st.session_state:
    try:
        r = requests.get(f"{API_BASE}/files", timeout=3)
        st.session_state.ingested_files = r.json().get("files", []) if r.status_code == 200 else []
    except Exception:
        st.session_state.ingested_files = []

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0





def get_health() -> dict:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def ingest_files(uploaded_files) -> dict:
    try:
        files = [
            ("files", (f.name, f.getvalue(), "application/pdf"))
            for f in uploaded_files
        ]
        r = requests.post(f"{API_BASE}/ingest", files=files, timeout=300)
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend. Is the FastAPI server running?"}
    except Exception as e:
        return {"error": str(e)}


def query_backend(question: str) -> dict:
    try:
        r = requests.post(
            f"{API_BASE}/query",
            json={"question": question},
            timeout=60,
        )
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend. Is the FastAPI server running?"}
    except Exception as e:
        return {"error": str(e)}


def _render_result(result: dict) -> None:
    """Render query result and append to chat history."""
    if "error" in result:
        answer = f"Error: {result['error']}"
        st.error(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        return

    answer = result.get("answer", "No answer returned.")
    st.markdown(answer)

    intent = result.get("intent", "")
    intent_colours = {
        "factual": "blue", "list": "green",
        "table": "orange", "chitchat": "gray", "refusal": "red",
    }
    colour = intent_colours.get(intent, "gray")
    st.markdown(f"**Intent:** :{colour}[{intent.upper()}]")

    citations = result.get("citations", [])
    if citations:
        with st.expander(f"📎 Sources ({len(citations)})"):
            for c in citations:
                st.markdown(
                    f"**{c['source_file']}** · Page {c['page_number']}\n\n"
                    f"> {c['excerpt']}"
                )

    if result.get("insufficient_evidence"):
        st.info("No sufficiently relevant content found in the knowledge base.")

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "meta": {
            "intent":                intent,
            "citations":             citations,
            "insufficient_evidence": result.get("insufficient_evidence", False),
        },
    })



with st.sidebar:
    health = get_health()
    if health:
        chunks = health.get('chunks_stored', 0)
        st.success(f"Backend online · {chunks} chunks stored")
    else:
        st.error("Backend offline")
        st.code("python3 -m uvicorn main:app --reload", language="bash")

    st.divider()

    st.subheader("Add to Knowledge Base")

    uploaded = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if st.button("Add to Knowledge Base", type="primary", use_container_width=True, key="add_kb"):
        if not uploaded:
            st.warning("Please select at least one PDF first.")
        else:
            files_to_ingest = list(uploaded)
            st.session_state.uploader_key += 1
            all_ok = True
            for f in files_to_ingest:
                with st.spinner(f"Ingesting {f.name}..."):
                    try:
                        r = requests.post(
                            f"{API_BASE}/ingest",
                            files=[("files", (f.name, f.getvalue(), "application/pdf"))],
                            timeout=300,
                        )
                        result = r.json()
                    except Exception as e:
                        result = {"error": str(e)}

                if "error" in result or "detail" in result:
                    st.error(f"{f.name}: {result.get('error') or result.get('detail')}")
                    all_ok = False
                else:
                    if f.name not in st.session_state.ingested_files:
                        st.session_state.ingested_files.append(f.name)

            st.rerun()

    if st.button("Clear Knowledge Base", type="primary", use_container_width=True, key="clear_kb"):
        try:
            r = requests.post(f"{API_BASE}/clear", timeout=10)
            if r.status_code == 200:
                st.session_state.ingested_files = []
                st.rerun()
            else:
                st.error("Failed to clear knowledge base.")
        except Exception as e:
            st.error(str(e))

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []

    st.divider()

    if st.session_state.ingested_files:
        st.subheader("Knowledge Base")
        for fname in list(st.session_state.ingested_files):
            col1, col2 = st.columns([5, 1])
            col1.markdown(f"📄 `{fname}`")
            if col2.button("✕", key=f"remove_{fname}", help=f"Remove {fname}"):
                try:
                    r = requests.post(f"{API_BASE}/remove", json={"filename": fname}, timeout=10)
                    if r.status_code == 200:
                        st.session_state.ingested_files.remove(fname)
                        st.rerun()
                    else:
                        st.error(f"Failed to remove {fname}")
                except Exception as e:
                    st.error(str(e))



st.title("StackAI RAG")
st.caption("Questions are answered using only the content of your uploaded PDFs. Sources are cited for every answer.")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "meta" in message:
            meta = message["meta"]
            intent = meta.get("intent", "")
            colour = {"factual": "blue", "list": "green", "table": "orange",
                      "chitchat": "gray", "refusal": "red"}.get(intent, "gray")
            st.markdown(f"**Intent:** :{colour}[{intent.upper()}]")

            citations = meta.get("citations", [])
            if citations:
                with st.expander(f"📎 Sources ({len(citations)})"):
                    for c in citations:
                        st.markdown(
                            f"**{c['source_file']}** · Page {c['page_number']}\n\n"
                            f"> {c['excerpt']}"
                        )

            if meta.get("insufficient_evidence"):
                st.info("No sufficiently relevant content found in the knowledge base.")

question = st.chat_input(
    "Ask a question about your documents...",
    disabled=not health,
)

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = query_backend(question)
        _render_result(result)

    st.rerun()
