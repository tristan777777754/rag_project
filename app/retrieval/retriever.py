"""
Hybrid Retriever: BM25 + Embedding + factor-domain query routing + cross-encoder

The key change in this version is that query routing is used to restrict the
candidate pool before scoring, instead of only applying soft boosts after a
mostly global retrieval step.
"""
from typing import List, Dict, Any, Optional

from app.embeddings.embedder import embed_query
from app.vector_store.chroma_store import get_collection, search_chunks_with_section_boost
from app.retrieval.query_expansion import expand_query
from app.retrieval.query_router import (
    classify_query,
    get_page_boost_for_query,
    get_query_route,
    get_section_boost_for_query,
    map_target_sections_to_canonical,
)
from app.retrieval.bm25_retriever import BM25Retriever

# Global BM25 retriever (lazy initialized)
_bm25_retriever = None


def _normalize_section(section: Optional[str]) -> str:
    return (section or "body").lower().strip()


def _load_all_chunks() -> List[Dict[str, Any]]:
    """Load all chunk documents/metadata from ChromaDB in a BM25-friendly format."""
    all_data = get_collection().get()
    chunks = []

    for doc, meta in zip(all_data.get("documents", []), all_data.get("metadatas", [])):
        meta = meta or {}
        chunks.append(
            {
                "text": doc,
                "page": meta.get("page", "Unknown"),
                "section": _normalize_section(meta.get("section", "body")),
                "position": meta.get("position", "body"),
            }
        )

    return chunks


def _get_bm25_retriever() -> BM25Retriever:
    """Lazy initialization of BM25 retriever from all chunks in ChromaDB."""
    global _bm25_retriever
    if _bm25_retriever is None:
        _bm25_retriever = BM25Retriever(_load_all_chunks())
    return _bm25_retriever


def _get_available_sections() -> List[str]:
    return sorted({_normalize_section(chunk.get("section")) for chunk in _load_all_chunks()})


def _filter_chunks_by_sections(chunks: List[Dict[str, Any]], allowed_sections: Optional[List[str]]) -> List[Dict[str, Any]]:
    if not allowed_sections:
        return [c for c in chunks if _normalize_section(c.get("section")) not in {"references", "acknowledgments"}]

    allowed = set(allowed_sections)
    filtered = [
        chunk for chunk in chunks
        if _normalize_section(chunk.get("section")) in allowed
    ]
    return filtered


def _retrieve_candidates_embedding(
    query: str,
    top_k: int = 20,
    allowed_sections: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Retrieve embedding candidates, then restrict them to routed sections when
    available. This keeps the implementation compatible with the current vector
    store, which does not yet expose metadata filtering at query time.
    """
    expanded_queries = expand_query(query)
    all_results = []
    seen_chunks = set()

    # Ask Chroma for a wider pool so post-filtering still leaves enough
    # candidates. When section filtering is active we query a bit deeper.
    raw_top_k = max(top_k * (5 if allowed_sections else 3), top_k)
    section_boost = get_section_boost_for_query(query, allowed_sections or _get_available_sections())

    for eq in expanded_queries[:3]:
        query_embedding = embed_query(eq)
        results = search_chunks_with_section_boost(
            query_embedding,
            top_k=raw_top_k,
            section_boost=section_boost,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for doc, meta in zip(documents, metadatas):
            doc_hash = hash(doc[:200])
            if doc_hash in seen_chunks:
                continue
            seen_chunks.add(doc_hash)

            meta = meta or {}
            result = {
                "text": doc,
                "page": meta.get("page", "Unknown"),
                "section": _normalize_section(meta.get("section", "body")),
                "position": meta.get("position", "body"),
            }
            all_results.append(result)

    filtered = _filter_chunks_by_sections(all_results, allowed_sections)
    return filtered[:top_k] if filtered else all_results[:top_k]


def _retrieve_candidates_bm25(
    query: str,
    top_k: int = 20,
    allowed_sections: Optional[List[str]] = None,
) -> List[Dict]:
    """Retrieve candidates using BM25, with optional section restriction."""
    try:
        bm25 = _get_bm25_retriever()
        # Get a deeper global pool, then filter by routed sections.
        raw_results = bm25.search(query, top_k=max(top_k * (5 if allowed_sections else 3), top_k))
        filtered = _filter_chunks_by_sections(raw_results, allowed_sections)
        return filtered[:top_k] if filtered else raw_results[:top_k]
    except Exception as e:
        print(f"BM25 search failed: {e}")
        return []


def _apply_query_reranking(results: List[Dict], query: str, available_sections: Optional[List[str]] = None) -> List[Dict]:
    """
    Re-rank results based on factor-specific query intent, section, and page
    relevance.
    """
    query_type = classify_query(query)
    section_boost = get_section_boost_for_query(query, available_sections or _get_available_sections())
    page_boost = get_page_boost_for_query(query)

    scored_results = []
    for chunk in results:
        section = _normalize_section(chunk.get("section", "body"))
        page = chunk.get("page", 999)
        base_score = chunk.get("hybrid_score", chunk.get("bm25_score", chunk.get("emb_score", 0.5)))

        sec_boost = section_boost.get(section, 1.0)
        pg_boost = page_boost.get(page, 1.0)
        final_score = base_score * sec_boost * pg_boost

        chunk_copy = chunk.copy()
        chunk_copy["section"] = section
        chunk_copy["final_score"] = final_score
        chunk_copy["query_type"] = query_type
        chunk_copy["boost_factors"] = f"sec:{sec_boost:.1f}, pg:{pg_boost:.1f}"
        scored_results.append(chunk_copy)

    scored_results.sort(key=lambda x: x["final_score"], reverse=True)
    return scored_results


def _merge_hybrid_results(
    embedding_results: List[Dict],
    bm25_results: List[Dict],
    alpha: float = 0.5,
) -> List[Dict]:
    """Merge embedding and BM25 results with weighted scoring."""
    merged = {}

    for rank, chunk in enumerate(embedding_results):
        chunk_id = hash(chunk["text"][:200])
        emb_score = 1.0 - (rank / len(embedding_results)) if embedding_results else 0
        merged[chunk_id] = {
            "chunk": chunk,
            "emb_score": emb_score,
            "bm25_score": 0,
        }

    if bm25_results:
        raw_scores = [c.get("bm25_score", 0) for c in bm25_results]
        max_score = max(raw_scores) if raw_scores else 1
        min_score = min(raw_scores) if raw_scores else 0
        score_range = max_score - min_score if max_score > min_score else 1

        for chunk in bm25_results:
            chunk_id = hash(chunk["text"][:200])
            raw_bm25 = chunk.get("bm25_score", 0)
            bm25_norm = (raw_bm25 - min_score) / score_range if score_range > 0 else 0

            if chunk_id in merged:
                merged[chunk_id]["bm25_score"] = bm25_norm
            else:
                merged[chunk_id] = {
                    "chunk": chunk,
                    "emb_score": 0,
                    "bm25_score": bm25_norm,
                }

    for data in merged.values():
        data["hybrid_score"] = alpha * data["bm25_score"] + (1 - alpha) * data["emb_score"]

    sorted_results = sorted(merged.values(), key=lambda x: x["hybrid_score"], reverse=True)

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
    hybrid_alpha: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant chunks using hybrid search with factor-investing-aware routing.

    The retriever now performs query-type-based candidate restriction before the
    main scoring stages, but falls back to broader retrieval if section routing
    produces too few candidates.
    """
    available_sections = _get_available_sections()
    route = get_query_route(query) if use_query_routing else {"query_type": "generic", "target_sections": []}
    allowed_sections = map_target_sections_to_canonical(route.get("target_sections", []), available_sections) if use_query_routing else None

    if use_query_routing:
        print(
            f"[Query Router] Type: {route['query_type']}, "
            f"Target: {route['target_sections']}, Canonical: {allowed_sections}"
        )

    # Larger candidate pool for the first pass. Query-aware filtering will prune
    # this down before reranking.
    initial_top_k = max(top_k * 4, 20)

    if use_hybrid:
        bm25_results = _retrieve_candidates_bm25(query, top_k=initial_top_k, allowed_sections=allowed_sections)
        emb_results = _retrieve_candidates_embedding(query, top_k=initial_top_k, allowed_sections=allowed_sections)
        candidates = _merge_hybrid_results(emb_results, bm25_results, alpha=hybrid_alpha)

        # Fallback to broader retrieval if section labels are noisy/sparse.
        if use_query_routing and len(candidates) < top_k:
            bm25_results = _retrieve_candidates_bm25(query, top_k=initial_top_k, allowed_sections=None)
            emb_results = _retrieve_candidates_embedding(query, top_k=initial_top_k, allowed_sections=None)
            candidates = _merge_hybrid_results(emb_results, bm25_results, alpha=hybrid_alpha)
    else:
        candidates = _retrieve_candidates_embedding(query, top_k=initial_top_k, allowed_sections=allowed_sections)
        if use_query_routing and len(candidates) < top_k:
            candidates = _retrieve_candidates_embedding(query, top_k=initial_top_k, allowed_sections=None)

    if use_query_routing:
        candidates = _apply_query_reranking(candidates, query, available_sections)

    if not use_reranker:
        return candidates[:top_k]

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
