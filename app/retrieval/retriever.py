from app.embeddings.embedder import embed_query
from app.vector_store.chroma_store import search_chunks_with_section_boost, search_chunks
from app.retrieval.query_expansion import expand_query, rewrite_for_abstract


def retrieve_relevant_chunks(query: str, top_k: int = 5, use_section_boost: bool = True):
    """
    Retrieve relevant chunks and their page numbers.
    Uses query expansion and multiple retrieval strategies for better coverage.
    
    Args:
        query: The search query
        top_k: Number of chunks to retrieve
        use_section_boost: Whether to boost abstract/intro sections
    """
    # Get expanded queries
    expanded_queries = expand_query(query)
    
    # Collect results from multiple query variations
    all_results = []
    seen_chunks = set()  # Track unique chunks by content hash
    
    for eq in expanded_queries[:3]:  # Use top 3 expanded queries
        query_embedding = embed_query(eq)
        
        if use_section_boost:
            results = search_chunks_with_section_boost(query_embedding, top_k=top_k * 2)
        else:
            results = search_chunks(query_embedding, top_k=top_k * 2)
        
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        
        for doc, meta in zip(documents, metadatas):
            # Deduplicate by content
            doc_hash = hash(doc[:200])  # Use first 200 chars as identifier
            if doc_hash in seen_chunks:
                continue
            seen_chunks.add(doc_hash)
            
            page = meta.get("page", "Unknown") if meta else "Unknown"
            section = meta.get("section", "body") if meta else "body"
            position = meta.get("position", "body") if meta else "body"
            
            # Score based on section priority
            priority_score = 0
            if section == "abstract":
                priority_score = 100
            elif section == "introduction":
                priority_score = 50
            elif section == "acknowledgments":
                priority_score = -100  # Deprioritize acknowledgments
            
            all_results.append({
                "text": doc,
                "page": page,
                "section": section,
                "position": position,
                "priority_score": priority_score
            })
    
    # Sort by priority (abstract/intro first), then take top_k
    all_results.sort(key=lambda x: x["priority_score"], reverse=True)
    
    # Return top_k, removing the priority_score field
    return [{k: v for k, v in r.items() if k != "priority_score"} for r in all_results[:top_k]]
