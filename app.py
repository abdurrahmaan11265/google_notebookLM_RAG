"""
NotebookLM Clone — RAG-powered document chat
"""

import streamlit as st
from rag_pipeline import index_document, answer_query

st.set_page_config(
    page_title="NotebookLM Clone",
    page_icon="📚",
    layout="wide",
)

st.title("📚 NotebookLM Clone")
st.caption("Upload a document and ask questions — answers are grounded in your file.")

# ── Session state ──────────────────────────────────────────────────────────────
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar — document upload ──────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Document")
    uploaded = st.file_uploader("Upload a PDF or .txt file", type=["pdf", "txt"])

    if uploaded:
        file_bytes = uploaded.read()
        if (
            st.session_state.doc_id is None
            or st.session_state.filename != uploaded.name
        ):
            with st.spinner("Processing document…"):
                doc_id, num_chunks = index_document(file_bytes, uploaded.name)
            st.session_state.doc_id = doc_id
            st.session_state.filename = uploaded.name
            st.session_state.num_chunks = num_chunks
            st.session_state.messages = []
            st.success(f"Indexed **{num_chunks}** chunks from _{uploaded.name}_")
        else:
            st.success(
                f"Using cached index — **{st.session_state.num_chunks}** chunks from _{st.session_state.filename}_"
            )

    if st.session_state.doc_id:
        st.divider()
        st.markdown("**Chunking strategy**")
        st.markdown(
            "Recursive Character Splitting  \n"
            "• chunk_size = 600 tokens  \n"
            "• chunk_overlap = 100 tokens  \n"
            "• separators: `\\n\\n` → `\\n` → `. ` → ` `"
        )
        st.divider()
        st.markdown("**Embedding model**  \n`all-MiniLM-L6-v2`")
        st.markdown("**Vector DB**  \nChromaDB (cosine similarity)")
        st.markdown("**LLM**  \nClaude claude-sonnet-4-6")

    if st.session_state.messages and st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ── Main — chat interface ──────────────────────────────────────────────────────
if not st.session_state.doc_id:
    st.info("Upload a document in the sidebar to get started.")
    st.stop()

st.subheader(f"Chat with: _{st.session_state.filename}_")

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("chunks"):
            with st.expander("View retrieved source chunks"):
                for i, chunk in enumerate(msg["chunks"], 1):
                    st.markdown(f"**Chunk {i}**")
                    st.text(chunk)

# Chat input
query = st.chat_input("Ask a question about your document…")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            result = answer_query(st.session_state.doc_id, query)
        st.markdown(result["answer"])
        if result["chunks"]:
            with st.expander("View retrieved source chunks"):
                for i, chunk in enumerate(result["chunks"], 1):
                    st.markdown(f"**Chunk {i}**")
                    st.text(chunk)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "chunks": result["chunks"],
        }
    )
