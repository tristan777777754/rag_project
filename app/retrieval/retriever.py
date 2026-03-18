"""
Hybrid Retriever: Combines BM25 (keyword) + Embedding (semantic) + Cross-encoder (reranking)
"""
from app.embeddings.embedder import embed_query
from app.vector_store.chroma_store import search_chunks_with_section_boost, search_chunks, get_collection_count
from app.retrieval.query_expansion import expand_query
from app.retrieval.bm25_retriever import BM25Retriever
from typing import List, Dict, Any

# Global BM25 retriever (lazy initialized)
_bm25_retriever = None

def _get_bm25_retriever() -> BM25Retriever:
    """Lazy initialization of BM25 retriever from all chunks in ChromaDB."""
    global _bm25_retriever
    if _bm25_retriever is None:
        # Get all chunks from ChromaDB
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


def _retrieve_candidates_embedding(query: str, top_k: int = 20, use_section_boost: bool = True) -> List[Dict]:
    """Retrieve candidates using embedding search."""
    expanded_queries = expand_query(query)
    all_results = []
    seen_chunks = set()
    
    for eq in expanded_queries[:2]:
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
            
            all_results.append({
                "text": doc,
                "page": page,
                "section": section,
                "position": position,
                "_hash": doc_hash
            })
    
    return all_results


def _retrieve_candidates_bm25(query: str, top_k: int = 20) -> List[Dict]:
    """Retrieve candidates using BM25 keyword search."""
    try:
        bm25 = _get_bm25_retriever()
        return bm25.search(query, top_k=top_k)
    except Exception as e:
        print(f"BM25 search failed: {e}")
        return []


def _merge_hybrid_results(
    embedding_results: List[Dict], 
    bm25_results: List[Dict], 
    alpha: float = 0.5
) -> List[Dict]:
    """
    Merge embedding and BM25 results with weighted scoring.
    
    Args:
        embedding_results: Results from embedding search
        bm25_results: Results from BM25 search
        alpha: Weight for BM25 (1-alpha for embedding)
    
    Returns:
        Merged and sorted results
    """
    merged = {}
    
    # Normalize embedding scores (rank-based)
    for rank, chunk in enumerate(embedding_results):
        chunk_id = chunk.get("_hash", hash(chunk["text"][:200]))
        # Higher rank = lower score (0 to 1 scale)
        emb_score = 1.0 - (rank / len(embedding_results)) if embedding_results else 0
        merged[chunk_id] = {
            "chunk": chunk,
            "emb_score": emb_score,
            "bm25_score": 0
        }
    
    # Normalize BM25 scores
    if bm25_results:
        # Get raw scores
        raw_scores = [c.get("bm25_score", 0) for c in bm25_results]
        max_score = max(raw_scores) if raw_scores else 1
        min_score = min(raw_scores) if raw_scores else 0
        score_range = max_score - min_score if max_score > min_score else 1
        
        for rank, chunk in enumerate(bm25_results):
            chunk_id = hash(chunk["text"][:200])
            # Normalize to 0-1
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
    
    # Return chunks without internal scores
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
    use_section_boost: bool = True, 
    use_reranker: bool = True,
    use_hybrid: bool = True,
    hybrid_alpha: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant chunks using hybrid search.
    
    Three-stage retrieval:
    1. Hybrid retrieval: BM25 + Embedding (recall)
    2. Cross-encoder reranking (precision)
    
    Args:
        query: The search query
        top_k: Number of chunks to retrieve
        use_section_boost: Whether to boost abstract/intro sections
        use_reranker: Whether to use cross-encoder reranking
        use_hybrid: Whether to use BM25 + Embedding hybrid
        hybrid_alpha: Weight for BM25 (0.5 = equal, higher = more BM25)
    """
    
    if use_hybrid:
        # Stage 1a: BM25 retrieval
        bm25_results = _retrieve_candidates_bm25(query, top_k=top_k * 2)
        
        # Stage 1b: Embedding retrieval
        emb_results = _retrieve_candidates_embedding(query, top_k=top_k * 2, use_section_boost=use_section_boost)
        
        # Stage 1c: Merge
        candidates = _merge_hybrid_results(emb_results, bm25_results, alpha=hybrid_alpha)
    else:
        # Fallback to embedding only
        candidates = _retrieve_candidates_embedding(query, top_k=top_k * 4, use_section_boost=use_section_boost)
    
    if not use_reranker:
        return candidates[:top_k]
    
    # Stage 2: Cross-encoder reranking
    try:
        from app.retrieval.reranker import rerank_chunks
        reranked = rerank_chunks(query, candidates, top_k=top_k)
        return reranked
    except Exception as e:
        print(f"Reranking failed: {e}, falling back to hybrid results")
        return candidates[:top_k]


def reset_retriever():
    """Reset the BM25 retriever (call when new PDF is ingested)."""
    global _bm25_retriever
    _bm25_retriever = None
