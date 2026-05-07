# NotebookLM Clone — RAG-Powered Document Chat

A full RAG pipeline that lets you upload any PDF or text file and have a grounded conversation with it, powered by Claude claude-sonnet-4-6.

## RAG Pipeline

```
Upload → Extract Text → Chunk → Embed → Store → Retrieve → Generate
```

| Stage | Tool / Approach |
|---|---|
| Ingestion | `pypdf` (PDF), plain UTF-8 (txt) |
| Chunking | Recursive Character Text Splitting (chunk_size=600, overlap=100) |
| Embedding | `all-MiniLM-L6-v2` via `sentence-transformers` |
| Vector DB | ChromaDB (local, persistent, cosine similarity) |
| Retrieval | Top-5 semantic search |
| Generation | Claude claude-sonnet-4-6 (context-only, no hallucination) |

## Chunking Strategy

**Recursive Character Text Splitting** splits text using a priority list of separators:
`"\n\n"` → `"\n"` → `". "` → `" "` → `""`

It tries the largest semantic boundary first (paragraph), falling back to finer splits only when needed. The 100-token overlap ensures no context is lost at chunk edges.

## Setup

```bash
pip install -r requirements.txt
# add ANTHROPIC_API_KEY to .env
streamlit run app.py
```

## Usage

1. Upload a PDF or `.txt` file in the sidebar
2. Wait for indexing to complete
3. Ask questions in the chat input
4. Expand "View retrieved source chunks" to see what context was used
