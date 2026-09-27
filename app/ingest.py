"""
Ingestion pipeline for the RAG Document Q&A API.

Usage:
    python -m app.ingest data/sample_document.txt
    python -m app.ingest data/sample_document.pdf

Supports .txt and .pdf files. Splits the document into overlapping
word-based chunks, embeds them with sentence-transformers, and stores
them in a persistent ChromaDB collection.
"""
import os
import sys
import uuid

import chromadb
from pypdf import PdfReader

from app.config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from app.embeddings import embed_texts


def extract_text(file_path: str) -> str:
    """Extract raw text from a .txt or .pdf file. Raises ValueError for other extensions."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    if ext == ".pdf":
        reader = PdfReader(file_path)
        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages_text.append(page_text)
        return "\n".join(pages_text)

    raise ValueError(f"Unsupported file type '{ext}'. Only .txt and .pdf are supported.")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks by word count.
    A simple, dependency-free splitter — good enough for short documents.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk.strip())
        if end >= len(words):
            break
        start = end - overlap  
    return chunks


def get_chroma_collection():
    """Get (or create) the persistent Chroma collection, using cosine distance."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def ingest_document(file_path: str) -> int:
    """Read a .txt or .pdf file, chunk it, embed the chunks, and store them. Returns chunk count."""
    text = extract_text(file_path)

    chunks = chunk_text(text)
    if not chunks:
        print("No extractable content found to ingest.")
        return 0

    embeddings = embed_texts(chunks)
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": file_path, "chunk_index": i} for i in range(len(chunks))]

    collection = get_chroma_collection()
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    print(f"Ingested {len(chunks)} chunks from '{file_path}' into collection '{COLLECTION_NAME}'.")
    return len(chunks)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m app.ingest <path_to_document.(txt|pdf)>")
        sys.exit(1)

    try:
        ingest_document(sys.argv[1])
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)