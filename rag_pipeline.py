"""
RAG Pipeline — ingestion → chunking → embedding → storage → retrieval → generation

Chunking Strategy: Recursive Character Text Splitting
  - Splits on ["\n\n", "\n", " ", ""] in order, preserving semantic boundaries
  - chunk_size=600, chunk_overlap=100
  - Overlap ensures context isn't lost at chunk boundaries
"""

import os
import uuid
import hashlib
from pathlib import Path

import anthropic
import chromadb
from chromadb.utils.embedding_functions import FastEmbedEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 5


def _get_chroma_collection(collection_name: str):
    embedding_fn = FastEmbedEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def _doc_collection_name(doc_id: str) -> str:
    return f"doc_{doc_id}"


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract plain text from a PDF or .txt file."""
    if filename.lower().endswith(".pdf"):
        from io import BytesIO
        reader = PdfReader(BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    else:
        return file_bytes.decode("utf-8", errors="replace")


def chunk_text(text: str) -> list[str]:
    """
    Recursive Character Text Splitting:
    Tries to split on paragraph breaks first, then newlines, then spaces,
    then individual characters — preserving as much semantic coherence as possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def index_document(file_bytes: bytes, filename: str) -> tuple[str, int]:
    """
    Full ingestion pipeline: extract → chunk → embed → store.
    Returns (doc_id, num_chunks).
    """
    doc_id = hashlib.md5(file_bytes).hexdigest()[:16]
    collection = _get_chroma_collection(_doc_collection_name(doc_id))

    if collection.count() > 0:
        return doc_id, collection.count()

    text = extract_text(file_bytes, filename)
    chunks = chunk_text(text)

    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

    collection.add(documents=chunks, ids=ids, metadatas=metadatas)

    return doc_id, len(chunks)


def retrieve_chunks(doc_id: str, query: str, top_k: int = TOP_K) -> list[str]:
    """Embed the query and retrieve the top-k most relevant chunks."""
    collection = _get_chroma_collection(_doc_collection_name(doc_id))
    results = collection.query(query_texts=[query], n_results=min(top_k, collection.count()))
    return results["documents"][0] if results["documents"] else []


def generate_answer(query: str, chunks: list[str]) -> str:
    """Use Claude claude-sonnet-4-6 to answer the query strictly from retrieved context."""
    context = "\n\n---\n\n".join(chunks)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    system_prompt = (
        "You are a document assistant. Answer the user's question using ONLY the context "
        "extracted from the uploaded document below. Do not use any outside knowledge. "
        "If the answer is not in the context, say: 'I could not find this information in the document.'\n\n"
        f"DOCUMENT CONTEXT:\n{context}"
    )

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": query}],
        system=system_prompt,
    )
    return message.content[0].text


def answer_query(doc_id: str, query: str) -> dict:
    """End-to-end retrieval + generation. Returns answer and source chunks."""
    chunks = retrieve_chunks(doc_id, query)
    if not chunks:
        return {"answer": "No relevant content found in the document.", "chunks": []}
    answer = generate_answer(query, chunks)
    return {"answer": answer, "chunks": chunks}
