"""
Retrieval logic for the RAG Document Q&A API.

Given a question, embeds it and queries ChromaDB for the top-k most
semantically similar chunks, converting Chroma's cosine *distance*
into a cosine *similarity* score (1 - distance) that's easier to reason about.
"""
from app.config import TOP_K
from app.embeddings import embed_query
from app.ingest import get_chroma_collection


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """
    Retrieve the top_k most relevant chunks for a question.

    Returns a list of dicts: {"text": str, "score": float, "metadata": dict}
    sorted by descending similarity score (0-1, higher = more similar).
    """
    collection = get_chroma_collection()

    if collection.count() == 0:
        return []

    query_embedding = embed_query(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )

    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    retrieved = []
    for doc, dist, meta in zip(documents, distances, metadatas):
        similarity = 1 - dist  
        retrieved.append({"text": doc, "score": round(similarity, 4), "metadata": meta})

    return retrieved
