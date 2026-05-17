"""
RAG Pipeline — ingestion → chunking → embedding → storage → retrieval → generation

Chunking Strategy: Recursive Character Text Splitting
  - Splits on ["\n\n", "\n", ". ", " ", ""] preserving semantic boundaries
  - chunk_size=600, chunk_overlap=100

Embedding Strategy: TF-IDF + Latent Semantic Analysis (LSA/SVD)
  - TfidfVectorizer (max 5000 features, sublinear TF scaling)
  - TruncatedSVD reduces to 128-dim dense vectors
  - L2-normalised for cosine similarity
  - Pure Python / numpy — no torch, no onnxruntime, ~30MB RAM

Query Improvement Techniques:
  1. Multi-Query Retrieval — Claude rewrites the query into N phrasings;
     chunks are retrieved for each and deduplicated by content.
  2. HyDE (Hypothetical Document Embeddings) — Claude generates a plausible
     passage that would answer the query; that passage is embedded and used
     for retrieval instead of the raw query, closing the query-document gap.
"""

import hashlib
import os

import anthropic
import chromadb
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 5
N_COMPONENTS = 128
MULTI_QUERY_N = 3  # extra query variants to generate

_embedders: dict = {}
_chroma_client = chromadb.EphemeralClient()


# ── Embedding ──────────────────────────────────────────────────────────────────

class LSAEmbedder:
    """TF-IDF + truncated SVD fitted on document chunks."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000, sublinear_tf=True, stop_words="english"
        )
        self.svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)

    def fit_transform(self, texts: list[str]) -> list[list[float]]:
        tfidf = self.vectorizer.fit_transform(texts)
        n = max(1, min(N_COMPONENTS, tfidf.shape[1] - 1, tfidf.shape[0] - 1))
        self.svd.n_components = n
        dense = self.svd.fit_transform(tfidf)
        return normalize(dense).tolist()

    def transform(self, texts: list[str]) -> list[list[float]]:
        tfidf = self.vectorizer.transform(texts)
        dense = self.svd.transform(tfidf)
        return normalize(dense).tolist()


# ── Storage helpers ────────────────────────────────────────────────────────────

def _collection(doc_id: str):
    return _chroma_client.get_or_create_collection(
        name=f"doc_{doc_id}",
        metadata={"hnsw:space": "cosine"},
    )


# ── Ingestion ──────────────────────────────────────────────────────────────────

def extract_text(file_bytes: bytes, filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        from io import BytesIO
        reader = PdfReader(BytesIO(file_bytes))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return file_bytes.decode("utf-8", errors="replace")


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def index_document(file_bytes: bytes, filename: str) -> tuple[str, int]:
    """Extract → chunk → embed (LSA) → store. Returns (doc_id, num_chunks)."""
    doc_id = hashlib.md5(file_bytes).hexdigest()[:16]
    col = _collection(doc_id)

    if col.count() > 0 and doc_id in _embedders:
        return doc_id, col.count()

    text = extract_text(file_bytes, filename)
    chunks = chunk_text(text)

    embedder = LSAEmbedder()
    embeddings = embedder.fit_transform(chunks)
    _embedders[doc_id] = embedder

    try:
        _chroma_client.delete_collection(f"doc_{doc_id}")
    except Exception:
        pass

    col = _collection(doc_id)
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]
    col.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)
    return doc_id, len(chunks)


# ── Query rewriting helpers ────────────────────────────────────────────────────

def _llm_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def rewrite_queries(query: str) -> list[str]:
    """
    Multi-Query: ask Claude to rephrase the query in MULTI_QUERY_N different ways.
    Returns the original query plus the generated variants.
    """
    prompt = (
        f"You are a query rewriting assistant helping improve document retrieval.\n"
        f"Rephrase the following question in {MULTI_QUERY_N} different ways that preserve "
        f"the original intent but use different vocabulary, structure, or perspective. "
        f"Output ONLY the rephrased questions, one per line, no numbering or extra text.\n\n"
        f"Original question: {query}"
    )
    msg = _llm_client().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    variants = [line.strip() for line in msg.content[0].text.strip().splitlines() if line.strip()]
    return [query] + variants[:MULTI_QUERY_N]


def generate_hypothetical_answer(query: str) -> str:
    """
    HyDE: ask Claude to write a short passage that *would* answer the query
    if it appeared in a document. The passage (not the query) is then embedded
    for retrieval — bridging the lexical gap between questions and answers.
    """
    prompt = (
        "Write a short, factual passage (2-4 sentences) that would directly answer "
        "the following question, as if it were excerpted from a relevant document. "
        "Do not mention that it is hypothetical — just write the passage.\n\n"
        f"Question: {query}"
    )
    msg = _llm_client().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


# ── Retrieval ──────────────────────────────────────────────────────────────────

def _retrieve_for_text(doc_id: str, text: str, top_k: int) -> list[str]:
    embedder = _embedders.get(doc_id)
    if not embedder:
        return []
    col = _collection(doc_id)
    embedding = embedder.transform([text])[0]
    results = col.query(
        query_embeddings=[embedding],
        n_results=min(top_k, col.count()),
    )
    return results["documents"][0] if results["documents"] else []


def retrieve_chunks(
    doc_id: str,
    query: str,
    top_k: int = TOP_K,
) -> tuple[list[str], list[str], str]:
    """
    Combined Multi-Query + HyDE retrieval.

    1. Generate MULTI_QUERY_N query variants via Claude.
    2. Generate a hypothetical answer passage via HyDE.
    3. Retrieve top-k chunks for each query variant AND for the HyDE passage.
    4. Deduplicate by exact content, preserving insertion order.

    Returns (chunks, all_queries_used, hypothetical_answer).
    """
    # Step 1 — Multi-Query
    queries = rewrite_queries(query)

    # Step 2 — HyDE
    hypothetical = generate_hypothetical_answer(query)

    # Step 3 — retrieve for every query + HyDE passage
    seen: set[str] = set()
    merged: list[str] = []

    search_texts = queries + [hypothetical]
    for text in search_texts:
        for chunk in _retrieve_for_text(doc_id, text, top_k):
            if chunk not in seen:
                seen.add(chunk)
                merged.append(chunk)

    # Keep at most top_k * 2 chunks so the context window stays reasonable
    return merged[: top_k * 2], queries, hypothetical


# ── Generation ─────────────────────────────────────────────────────────────────

def generate_answer(query: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(chunks)
    message = _llm_client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=(
            "You are a document assistant. Answer using ONLY the context below from the "
            "uploaded document. Do not use outside knowledge. "
            "If the answer isn't in the context, say: "
            "'I could not find this information in the document.'\n\n"
            f"DOCUMENT CONTEXT:\n{context}"
        ),
        messages=[{"role": "user", "content": query}],
    )
    return message.content[0].text


# ── Public entry point ─────────────────────────────────────────────────────────

def answer_query(doc_id: str, query: str) -> dict:
    """End-to-end retrieval + generation with multi-query + HyDE."""
    chunks, queries_used, hypothetical = retrieve_chunks(doc_id, query)
    if not chunks:
        return {
            "answer": "No relevant content found in the document.",
            "chunks": [],
            "queries_used": [query],
            "hypothetical": "",
        }
    return {
        "answer": generate_answer(query, chunks),
        "chunks": chunks,
        "queries_used": queries_used,
        "hypothetical": hypothetical,
    }
