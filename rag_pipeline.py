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
"""

import hashlib
import os

import anthropic
import chromadb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 5
N_COMPONENTS = 128

# module-level cache: doc_id → fitted LSAEmbedder
_embedders: dict = {}


class LSAEmbedder:
    """Fit TF-IDF + SVD on document chunks, then transform queries in the same space."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, sublinear_tf=True, stop_words="english")
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


def _get_embedder(doc_id: str) -> LSAEmbedder | None:
    return _embedders.get(doc_id)


def _get_collection(doc_id: str):
    client = chromadb.EphemeralClient()
    return client.get_or_create_collection(
        name=f"doc_{doc_id}",
        metadata={"hnsw:space": "cosine"},
    )


# persistent chroma client shared across calls
_chroma_client = chromadb.EphemeralClient()


def _collection(doc_id: str):
    return _chroma_client.get_or_create_collection(
        name=f"doc_{doc_id}",
        metadata={"hnsw:space": "cosine"},
    )


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
    """Extract → chunk → embed (LSA) → store in ChromaDB. Returns (doc_id, num_chunks)."""
    doc_id = hashlib.md5(file_bytes).hexdigest()[:16]

    col = _collection(doc_id)
    if col.count() > 0 and doc_id in _embedders:
        return doc_id, col.count()

    text = extract_text(file_bytes, filename)
    chunks = chunk_text(text)

    embedder = LSAEmbedder()
    embeddings = embedder.fit_transform(chunks)
    _embedders[doc_id] = embedder

    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

    try:
        _chroma_client.delete_collection(f"doc_{doc_id}")
    except Exception:
        pass

    col = _collection(doc_id)
    col.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)
    return doc_id, len(chunks)


def retrieve_chunks(doc_id: str, query: str, top_k: int = TOP_K) -> list[str]:
    embedder = _embedders.get(doc_id)
    if not embedder:
        return []
    col = _collection(doc_id)
    query_embedding = embedder.transform([query])[0]
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, col.count()),
    )
    return results["documents"][0] if results["documents"] else []


def generate_answer(query: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(chunks)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=(
            "You are a document assistant. Answer using ONLY the context below from the uploaded document. "
            "Do not use outside knowledge. If the answer isn't in the context, say: "
            "'I could not find this information in the document.'\n\n"
            f"DOCUMENT CONTEXT:\n{context}"
        ),
        messages=[{"role": "user", "content": query}],
    )
    return message.content[0].text


def answer_query(doc_id: str, query: str) -> dict:
    chunks = retrieve_chunks(doc_id, query)
    if not chunks:
        return {"answer": "No relevant content found in the document.", "chunks": []}
    return {"answer": generate_answer(query, chunks), "chunks": chunks}
