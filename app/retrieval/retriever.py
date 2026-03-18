from app.embeddings.embedder import embed_query
from app.vector_store.chroma_store import search_chunks_with_section_boost, search_chunks
from app.retrieval.query_expansion import expand_query, rewrite_for_abstract


def _retrieve_candidates(query: str, top_k: int = 20, use_section_boost: bool = True) -> list:
    """
    Internal function: retrieve candidate chunks using embeddings.
    Returns more results for reranking.
    """
    # Get expanded queries
    expanded_queries = expand_query(query)
    
    # Collect results from multiple query variations
    all_results = []
    seen_chunks = set()
    
    for eq in expanded_queries[:2]:  # Use top 2 expanded queries for speed
        query_embedding = embed_query(eq)
        
        if use_section_boost:
            results = search_chunks_with_section_boost(query_embedding, top_k=top_k)
        else:
            results = search_chunks(query_embedding, top_k=top_k)
        
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        
        for doc, meta in zip(documents, metadatas):
            doc_hash = hash(doc[:200])
            if doc_hash in seen_chunks:
                continue
            seen_chunks.add(doc_hash)
            
            page = meta.get("page", "Unknown") if meta else "Unknown"
            section = meta.get("section", "body") if meta else "body"
            position = meta.get("position", "body") if meta else "body"
            
            priority_score = 0
            if section == "abstract":
                priority_score = 100
            elif section == "introduction":
                priority_score = 50
            elif section == "acknowledgments":
                priority_score = -100
            
            all_results.append({
                "text": doc,
                "page": page,
                "section": section,
                "position": position,
                "priority_score": priority_score
            })
    
    all_results.sort(key=lambda x: x["priority_score"], reverse=True)
    return [{k: v for k, v in r.items() if k != "priority_score"} for r in all_results]


def retrieve_relevant_chunks(query: str, top_k: int = 5, use_section_boost: bool = True, use_reranker: bool = True):
    """
    Retrieve relevant chunks and their page numbers.
    
    Two-stage retrieval:
    1. Embedding-based retrieval (with query expansion and section boosting)
    2. Cross-encoder reranking (optional, for better relevance)
    
    Args:
        query: The search query
        top_k: Number of chunks to retrieve
        use_section_boost: Whether to boost abstract/intro sections
        use_reranker: Whether to use cross-encoder reranking
    """
    # Stage 1: Get candidates
    candidates = _retrieve_candidates(query, top_k=top_k * 4, use_section_boost=use_section_boost)
    
    if not use_reranker:
        return candidates[:top_k]
    
    # Stage 2: Cross-encoder reranking
    try:
        from app.retrieval.reranker import rerank_chunks
        reranked = rerank_chunks(query, candidates, top_k=top_k)
        return reranked
    except Exception as e:
        print(f"Reranking failed: {e}, falling back to embedding results")
        return candidates[:top_k]
