from app.embeddings.embedder import embed_query
from app.vector_store.chroma_store import search_chunks


def retrieve_relevant_chunks(query: str, top_k: int = 5):
    """
    Retrieve relevant chunks and their page numbers.
    """

    query_embedding = embed_query(query)
    results = search_chunks(query_embedding, top_k=top_k)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    chunks = []

    for doc, meta in zip(documents, metadatas):
        page = meta["page"] if meta is not None and "page" in meta else "Unknown"
        chunks.append({
            "text": doc,
            "page": page
        })

    return chunks