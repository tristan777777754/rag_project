"""
Hybrid Retriever: Combines BM25 (keyword) + Embedding (semantic) + Query Routing + Cross-encoder
"""
from app.embeddings.embedder import embed_query
from app.vector_store.chroma_store import search_chunks_with_section_boost, search_chunks, get_collection_count
from app.retrieval.query_expansion import expand_query
from app.retrieval.query_router import get_section_boost_for_query, get_page_boost_for_query, classify_query
from app.retrieval.bm25_retriever import BM25Retriever
from typing import List, Dict, Any

# Global BM25 retriever (lazy initialized)
_bm25_retriever = None

def _get_bm25_retriever() -> BM25Retriever:
    """Lazy initialization of BM25 retriever from all chunks in ChromaDB."""
    global _bm25_retriever
    if _bm25_retriever is None:
        from app.vector_store.chroma_store import collection
        all_data = collection.get()
        
        chunks = []
        for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
            chunks.append({
                "text": doc,
                "page": meta.get("page", "Unknown") if meta else "Unknown",
                "section": meta.get("section", "body") if meta else "body",
                "position": meta.get("position", "body") if meta else "body"
            })
        
        _bm25_retriever = BM25Retriever(chunks)
    return _bm25_retriever


def _retrieve_candidates_embedding(query: str, top_k: int = 20, use_query_routing: bool = False) -> List[Dict]:
    """
    Retrieve candidates using embedding search.
    Note: Query routing is applied AFTER hybrid merge, not here.
    """
    expanded_queries = expand_query(query)
    all_results = []
    seen_chunks = set()
    
    # Use default section boost (no query routing at this stage)
    section_boost = None
    
    for eq in expanded_queries[:2]:
        query_embedding = embed_query(eq)
        
        results = search_chunks_with_section_boost(
            query_embedding, 
            top_k=top_k,
            section_boost=section_boost
        )
        
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
            
            all_results.append({
                "text": doc,
                "page": page,
                "section": section,
                "position": position,
                "_hash": doc_hash
            })
    
    return [{k: v for k, v in r.items() if k != "_hash"} for r in all_results]


def _retrieve_candidates_bm25(query: str, top_k: int = 20) -> List[Dict]:
    """Retrieve candidates using BM25 keyword search."""
    try:
        bm25 = _get_bm25_retriever()
        return bm25.search(query, top_k=top_k)
    except Exception as e:
        print(f"BM25 search failed: {e}")
        return []


def _apply_query_reranking(results: List[Dict], query: str) -> List[Dict]:
    """
    Re-rank results based on query type, section, and page relevance.
    """
    query_type = classify_query(query)
    section_boost = get_section_boost_for_query(query)
    page_boost = get_page_boost_for_query(query)
    
    scored_results = []
    for chunk in results:
        section = chunk.get("section", "body")
        page = chunk.get("page", 999)
        base_score = chunk.get("hybrid_score", 0.5)
        
        # Apply section boost
        sec_boost = section_boost.get(section, 1.0)
        
        # Apply page boost (for introduction queries, boost early pages)
        pg_boost = page_boost.get(page, 1.0)
        
        # Combined boost
        final_score = base_score * sec_boost * pg_boost
        
        chunk_copy = chunk.copy()
        chunk_copy["final_score"] = final_score
        chunk_copy["query_type"] = query_type
        chunk_copy["boost_factors"] = f"sec:{sec_boost:.1f}, pg:{pg_boost:.1f}"
        scored_results.append(chunk_copy)
    
    # Sort by final score
    scored_results.sort(key=lambda x: x["final_score"], reverse=True)
    return scored_results


def _merge_hybrid_results(
    embedding_results: List[Dict], 
    bm25_results: List[Dict], 
    alpha: float = 0.5
) -> List[Dict]:
    """Merge embedding and BM25 results with weighted scoring."""
    merged = {}
    
    # Normalize embedding scores (rank-based)
    for rank, chunk in enumerate(embedding_results):
        chunk_id = hash(chunk["text"][:200])
        emb_score = 1.0 - (rank / len(embedding_results)) if embedding_results else 0
        merged[chunk_id] = {
            "chunk": chunk,
            "emb_score": emb_score,
            "bm25_score": 0
        }
    
    # Normalize BM25 scores
    if bm25_results:
        raw_scores = [c.get("bm25_score", 0) for c in bm25_results]
        max_score = max(raw_scores) if raw_scores else 1
        min_score = min(raw_scores) if raw_scores else 0
        score_range = max_score - min_score if max_score > min_score else 1
        
        for rank, chunk in enumerate(bm25_results):
            chunk_id = hash(chunk["text"][:200])
            raw_bm25 = chunk.get("bm25_score", 0)
            bm25_norm = (raw_bm25 - min_score) / score_range if score_range > 0 else 0
            
            if chunk_id in merged:
                merged[chunk_id]["bm25_score"] = bm25_norm
            else:
                merged[chunk_id] = {
                    "chunk": chunk,
                    "emb_score": 0,
                    "bm25_score": bm25_norm
                }
    
    # Calculate hybrid score
    for chunk_id, data in merged.items():
        data["hybrid_score"] = alpha * data["bm25_score"] + (1 - alpha) * data["emb_score"]
    
    # Sort by hybrid score
    sorted_results = sorted(merged.values(), key=lambda x: x["hybrid_score"], reverse=True)
    
    # Return chunks with scores
    final_results = []
    for data in sorted_results:
        chunk = data["chunk"].copy()
        chunk["hybrid_score"] = data["hybrid_score"]
        chunk["emb_score"] = data["emb_score"]
        chunk["bm25_score"] = data["bm25_score"]
        final_results.append(chunk)
    
    return final_results


def retrieve_relevant_chunks(
    query: str, 
    top_k: int = 5, 
    use_query_routing: bool = True,
    use_reranker: bool = True,
    use_hybrid: bool = True,
    hybrid_alpha: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant chunks using hybrid search with query routing.
    
    Args:
        query: The search query
        top_k: Number of chunks to retrieve
        use_query_routing: Whether to use query-aware section routing
        use_reranker: Whether to use cross-encoder reranking
        use_hybrid: Whether to use BM25 + Embedding hybrid
        hybrid_alpha: Weight for BM25 (0.5 = equal)
    """
    
    # Log query classification
    if use_query_routing:
        query_type = classify_query(query)
        # Detect available sections from a sample of candidates
        sample_results = _retrieve_candidates_embedding(query, top_k=10, use_query_routing=False)
        available_sections = list(set(c.get('section', 'body') for c in sample_results))
        section_boost = get_section_boost_for_query(query, available_sections)
        print(f"[Query Router] Type: {query_type}, Available: {available_sections}, Boost: {section_boost}")
    
    # Use larger initial top_k to ensure early pages are recalled
    # Then query routing will boost them appropriately
    initial_top_k = max(top_k * 4, 20)  # At least 20 to capture introduction content
    
    if use_hybrid:
        # Stage 1: Hybrid retrieval
        bm25_results = _retrieve_candidates_bm25(query, top_k=initial_top_k)
        emb_results = _retrieve_candidates_embedding(query, top_k=initial_top_k, use_query_routing=False)
        candidates = _merge_hybrid_results(emb_results, bm25_results, alpha=hybrid_alpha)
    else:
        candidates = _retrieve_candidates_embedding(query, top_k=initial_top_k, use_query_routing=False)
    
    # Stage 2: Query-aware re-ranking
    if use_query_routing:
        candidates = _apply_query_reranking(candidates, query)
    
    if not use_reranker:
        return candidates[:top_k]
    
    # Stage 3: Cross-encoder reranking
    try:
        from app.retrieval.reranker import rerank_chunks
        reranked = rerank_chunks(query, candidates, top_k=top_k)
        return reranked
    except Exception as e:
        print(f"Reranking failed: {e}")
        return candidates[:top_k]


def reset_retriever():
    """Reset the BM25 retriever."""
    global _bm25_retriever
    _bm25_retriever = None
