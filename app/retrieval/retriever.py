from app.embeddings.embedder import embed_query
from app.vector_store.chroma_store import search_chunks_with_section_boost


def retrieve_relevant_chunks(query: str, top_k: int = 5, use_section_boost: bool = True):
    """
    Retrieve relevant chunks and their page numbers.
    
    Args:
        query: The search query
        top_k: Number of chunks to retrieve
        use_section_boost: Whether to boost abstract/intro sections
    """
    query_embedding = embed_query(query)
    
    if use_section_boost:
        results = search_chunks_with_section_boost(query_embedding, top_k=top_k)
    else:
        # Fallback to basic search if needed
        from app.vector_store.chroma_store import search_chunks
        results = search_chunks(query_embedding, top_k=top_k)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    chunks = []

    for doc, meta in zip(documents, metadatas):
        page = meta.get("page", "Unknown") if meta else "Unknown"
        section = meta.get("section", "body") if meta else "body"
        position = meta.get("position", "body") if meta else "body"
        
        chunks.append({
            "text": doc,
            "page": page,
            "section": section,
            "position": position
        })

    return chunks
