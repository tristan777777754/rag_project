"""
Cross-Encoder Re-ranker for improved retrieval relevance.
Uses a lightweight cross-encoder to score query-document pairs.
"""
from sentence_transformers import CrossEncoder
import numpy as np
from typing import List, Dict, Any

# Use a lightweight cross-encoder model
# This is a small model that runs fast on CPU
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Lazy load the model
_reranker = None

def get_reranker():
    """Lazy load the cross-encoder model."""
    global _reranker
    if _reranker is None:
        print(f"Loading cross-encoder model: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank_chunks(query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Re-rank chunks using cross-encoder for better relevance.
    
    Args:
        query: The search query
        chunks: List of chunk dicts with 'text', 'page', 'section'
        top_k: Number of top chunks to return after reranking
    
    Returns:
        Re-ranked chunks sorted by cross-encoder score
    """
    if not chunks:
        return []
    
    reranker = get_reranker()
    
    # Prepare query-chunk pairs
    pairs = [(query, chunk["text"]) for chunk in chunks]
    
    # Get cross-encoder scores
    # Score range: typically -10 to 10, higher is more relevant
    scores = reranker.predict(pairs, show_progress_bar=False)
    
    # Attach scores to chunks
    scored_chunks = []
    for chunk, score in zip(chunks, scores):
        scored_chunks.append({
            **chunk,
            "rerank_score": float(score),
            "retrieval_method": "cross_encoder"
        })
    
    # Sort by rerank score (descending)
    scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    return scored_chunks[:top_k]


def hybrid_retrieve_and_rerank(
    query: str,
    initial_retrieval_fn,
    initial_top_k: int = 20,
    final_top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Two-stage retrieval: embedding retrieval + cross-encoder reranking.
    
    Args:
        query: Search query
        initial_retrieval_fn: Function to get initial candidates (e.g., retrieve_relevant_chunks)
        initial_top_k: Number of candidates to retrieve initially
        final_top_k: Number of chunks to return after reranking
    """
    # Stage 1: Get candidates using embedding retrieval
    candidates = initial_retrieval_fn(query, top_k=initial_top_k)
    
    # Stage 2: Re-rank using cross-encoder
    reranked = rerank_chunks(query, candidates, top_k=final_top_k)
    
    return reranked
