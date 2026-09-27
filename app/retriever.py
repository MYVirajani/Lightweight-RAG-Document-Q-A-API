from app.config import TOP_K
from app.embeddings import embed_query
from app.ingest import get_chroma_collection


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    
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
