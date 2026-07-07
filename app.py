"""
app.py — Streamlit UI for the "chat with your documents" assistant.

Run it (with Ollama running and the models pulled):

    streamlit run app.py

Upload PDFs or text files in the sidebar, index them, then ask questions.
Every answer shows the sources it was drawn from.
"""

import os
import tempfile

import streamlit as st

from rag import RAGPipeline


st.set_page_config(page_title="Chat with your documents", page_icon="📄", layout="wide")


# ---- session state (persists across Streamlit reruns) -----------------
if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline()
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---- sidebar: upload & index ------------------------------------------
with st.sidebar:
    st.header("📄 Your documents")

    uploaded = st.file_uploader(
        "Upload PDFs or text files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded:
        new_files = [f for f in uploaded if f.name not in st.session_state.indexed_files]

        if new_files and st.button(f"Index {len(new_files)} new file(s)"):
            with st.spinner("Reading and indexing..."):
                tmpdir = tempfile.mkdtemp()
                paths = []
                for f in new_files:
                    path = os.path.join(tmpdir, f.name)
                    with open(path, "wb") as out:
                        out.write(f.getbuffer())
                    paths.append(path)

                added = st.session_state.rag.add_documents(paths)
                st.session_state.chunk_count += added
                st.session_state.indexed_files.extend(f.name for f in new_files)

            st.success(f"Indexed {added} chunks.")

    if st.session_state.indexed_files:
        st.caption("Indexed documents:")
        for name in st.session_state.indexed_files:
            st.caption(f"• {name}")
        st.caption(f"**{st.session_state.chunk_count}** chunks total")

    st.divider()
    st.caption("🔒 Runs locally on Ollama — your documents never leave your machine.")


# ---- main: chat -------------------------------------------------------
st.title("Chat with your documents")

if not st.session_state.indexed_files:
    st.info("Upload one or more documents in the sidebar and click **Index** to begin.")
else:
    # replay the conversation so far
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.caption(f"[{s['n']}] {s['source']}")

    prompt = st.chat_input("Ask a question about your documents...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching your documents..."):
                result = st.session_state.rag.query(prompt)
            st.write(result["answer"])
            if result["sources"]:
                with st.expander("Sources"):
                    for s in result["sources"]:
                        st.caption(f"[{s['n']}] {s['source']}")

        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
        )
