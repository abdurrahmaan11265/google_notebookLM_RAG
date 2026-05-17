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
for key, default in [
    ("doc_id", None),
    ("filename", None),
    ("num_chunks", 0),
    ("messages", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ────────────────────────────────────────────────────────────────────
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
                f"Using cached index — **{st.session_state.num_chunks}** chunks "
                f"from _{st.session_state.filename}_"
            )

    if st.session_state.doc_id:
        st.divider()
        st.markdown("**Pipeline**")
        st.markdown(
            "**Chunking:** Recursive Character Splitting  \n"
            "• chunk_size = 600 · overlap = 100\n\n"
            "**Embedding:** TF-IDF + LSA (128-dim)  \n"
            "**Vector DB:** ChromaDB (cosine)  \n"
            "**LLM:** Claude claude-sonnet-4-6\n\n"
            "**Query techniques:**  \n"
            "• Multi-Query (3 rephrases)  \n"
            "• HyDE (hypothetical passage)"
        )
        st.divider()

    if st.session_state.messages and st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ── Main chat ──────────────────────────────────────────────────────────────────
if not st.session_state.doc_id:
    st.info("Upload a document in the sidebar to get started.")
    st.stop()

st.subheader(f"Chat with: _{st.session_state.filename}_")

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("queries_used") or msg.get("hypothetical"):
                with st.expander("Query rewriting details"):
                    if msg.get("queries_used"):
                        st.markdown("**Multi-Query variants:**")
                        for i, q in enumerate(msg["queries_used"]):
                            label = "Original" if i == 0 else f"Variant {i}"
                            st.markdown(f"- `{label}`: {q}")
                    if msg.get("hypothetical"):
                        st.markdown("**HyDE hypothetical passage:**")
                        st.info(msg["hypothetical"])
            if msg.get("chunks"):
                with st.expander("Retrieved source chunks"):
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
        with st.spinner("Rewriting queries · retrieving · generating…"):
            result = answer_query(st.session_state.doc_id, query)

        st.markdown(result["answer"])

        with st.expander("Query rewriting details"):
            st.markdown("**Multi-Query variants:**")
            for i, q in enumerate(result.get("queries_used", [query])):
                label = "Original" if i == 0 else f"Variant {i}"
                st.markdown(f"- `{label}`: {q}")
            if result.get("hypothetical"):
                st.markdown("**HyDE hypothetical passage:**")
                st.info(result["hypothetical"])

        if result["chunks"]:
            with st.expander("Retrieved source chunks"):
                for i, chunk in enumerate(result["chunks"], 1):
                    st.markdown(f"**Chunk {i}**")
                    st.text(chunk)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "chunks": result["chunks"],
            "queries_used": result.get("queries_used", []),
            "hypothetical": result.get("hypothetical", ""),
        }
    )
