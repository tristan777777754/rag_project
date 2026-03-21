import chromadb

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "pdf_chunks"

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = None


def get_collection():
    global collection
    if collection is None:
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def store_chunks(chunks, embeddings):
    """
    Store chunk text, embeddings, and page/section metadata in ChromaDB.
    """
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [chunk["text"] for chunk in chunks]
    
    # Store extended metadata including section and position
    metadatas = []
    for chunk in chunks:
        meta = {
            "page": chunk["page"],
            "section": chunk.get("section", "body"),
            "position": chunk.get("position", "body")
        }
        metadatas.append(meta)

    get_collection().add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )


def search_chunks(query_embedding: list[float], top_k: int = 5):
    results = get_collection().query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results


def search_chunks_with_section_boost(
    query_embedding: list[float], 
    top_k: int = 5,
    section_boost: dict = None
):
    """
    Search chunks with optional section-based boosting.
    Returns more results than needed, then re-ranks by section priority.
    """
    if section_boost is None:
        section_boost = {
            "abstract": 1.5,
            "introduction": 1.3,
            "methodology": 1.1,
            "results": 1.1,
            "conclusion": 1.0,
            "body": 1.0,
            "references": 0.5
        }
    
    # Get more results to allow for re-ranking
    results = get_collection().query(
        query_embeddings=[query_embedding],
        n_results=top_k * 3  # Get 3x to allow filtering/reranking
    )
    
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0] if "distances" in results else [1.0] * len(documents)
    embeddings = results.get("embeddings", [None] * len(documents))
    
    # Calculate boosted scores
    scored_results = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        section = meta.get("section", "body") if meta else "body"
        boost = section_boost.get(section, 1.0)
        
        # Convert distance to similarity (lower distance = higher similarity)
        # Chroma returns L2 distance by default
        base_score = 1.0 / (1.0 + dist) if dist is not None else 0.5
        boosted_score = base_score * boost
        
        scored_results.append({
            "document": doc,
            "metadata": meta,
            "score": boosted_score,
            "original_score": base_score
        })
    
    # Sort by boosted score and take top_k
    scored_results.sort(key=lambda x: x["score"], reverse=True)
    top_results = scored_results[:top_k]
    
    # Format to match original return structure
    return {
        "documents": [[r["document"] for r in top_results]],
        "metadatas": [[r["metadata"] for r in top_results]],
        "distances": [[1.0 - r["score"] for r in top_results]]  # Approximate distance
    }


def get_collection_count():
    return get_collection().count()


def reset_collection():
    """Delete and recreate the collection for fresh experiments."""
    global collection
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection
