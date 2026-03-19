"""
Cross-Encoder Re-ranker for improved retrieval relevance.
Uses a lightweight cross-encoder to score query-document pairs.
"""
from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

# Use a lightweight cross-encoder model.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Lazy load the model.
_reranker = None


def get_reranker():
    """Lazy load the cross-encoder model."""
    global _reranker
    if _reranker is None:
        print(f"Loading cross-encoder model: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def _format_chunk_for_reranker(chunk: Dict[str, Any]) -> str:
    """
    Inject lightweight structure metadata so the reranker can reason about
    section and page context, not just raw text.
    """
    section = chunk.get("section", "body")
    page = chunk.get("page", "Unknown")
    text = chunk.get("text", "")
    return f"[Section: {section}]\n[Page: {page}]\n{text}"


def rerank_chunks(query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Re-rank chunks using a cross-encoder for better precision.

    Args:
        query: The search query
        chunks: List of chunk dicts with 'text', 'page', 'section'
        top_k: Number of top chunks to return after reranking
    """
    if not chunks:
        return []

    reranker = get_reranker()

    pairs = [(query, _format_chunk_for_reranker(chunk)) for chunk in chunks]
    scores = reranker.predict(pairs, show_progress_bar=False)

    scored_chunks = []
    for chunk, score in zip(chunks, scores):
        scored_chunks.append(
            {
                **chunk,
                "rerank_score": float(score),
                "rerank_input": _format_chunk_for_reranker(chunk),
                "retrieval_method": "cross_encoder",
            }
        )

    scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored_chunks[:top_k]


def hybrid_retrieve_and_rerank(
    query: str,
    initial_retrieval_fn,
    initial_top_k: int = 20,
    final_top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Two-stage retrieval: initial retrieval + cross-encoder reranking."""
    candidates = initial_retrieval_fn(query, top_k=initial_top_k)
    return rerank_chunks(query, candidates, top_k=final_top_k)
