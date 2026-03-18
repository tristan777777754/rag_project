# Iteration 3: Cross-Encoder Re-ranking

## Date
2026-03-18

## Problem Identified
While hit rate was 100%, content **relevance** was weak:
- Q2 "What problem does this paper solve?" → relevance score **0.0** (terrible!)
- Retrieved page 1 chunks, but not the ones discussing "the problem"

Root cause: Bi-encoder (embedding) retrieval only measures semantic similarity, not actual relevance to the question.

## Solution: Two-Stage Retrieval

### Stage 1: Embedding Retrieval (Recall)
- Get 20 candidate chunks using semantic similarity
- Fast, captures broad context

### Stage 2: Cross-Encoder Re-ranking (Precision)
- Use `cross-encoder/ms-marco-MiniLM-L-6-v2` to score [query, chunk] pairs
- Cross-attention between query and document = much better relevance
- Return top 5 re-ranked chunks

## Implementation

### New File: `app/retrieval/reranker.py`
```python
# Load cross-encoder model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Score query-document pairs
pairs = [(query, chunk["text"]) for chunk in candidates]
scores = reranker.predict(pairs)
```

### Modified: `app/retrieval/retriever.py`
- Added `_retrieve_candidates()` for stage 1
- Modified `retrieve_relevant_chunks()` to optionally use reranker
- `use_reranker=True` by default

## Results

| Metric | Iter 2 | Iter 3 | Change |
|--------|--------|--------|--------|
| **Avg Score** | 0.79 | **0.85** | +0.06 |
| Hit Rate | 100% | 90% | -10% |
| Q2 Relevance | 0.0 | **0.80** | 🔥 +0.80! |
| Q1 Score | 0.78 | **1.00** | Perfect! |
| Q3 Score | 0.78 | **1.00** | Perfect! |

### Specific Improvements

**Q2: "What problem does this paper solve?"**
- Before: Retrieved abstract chunks not mentioning "problem"
- After: Retrieved chunks specifically about HJM computational challenges
- Relevance: 0.0 → **0.80**

**Q1: "What does this paper introduce?"**
- Score: 0.78 → **1.00** (perfect)

### Trade-off
- Q10 (auto differentiation) hit rate dropped - was 100% in iter 2
- But overall score still improved due to better relevance elsewhere

## Model Details

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Size: ~80MB (lightweight)
- Speed: Fast on CPU (~50ms per query)
- Trained on: Microsoft MARCO (passage ranking)
- Score range: -10 to +10 (higher = more relevant)

## Decision
✅ **KEEP** - Score improved from 0.79 → 0.85, and relevance scores are much better.

Hit rate trade-off is acceptable because the retrieved content is actually relevant to the questions now.

## Files Modified
- `app/retrieval/reranker.py` - New cross-encoder module
- `app/retrieval/retriever.py` - Two-stage retrieval with reranking
- `evaluation/run_iteration_3.py` - Evaluation script

## Next Steps
Iteration 4 ideas:
- Fix Q10 miss (query expansion for "auto differentiation")
- Hybrid retrieval (BM25 + embeddings) for better keyword matching
- Contextual compression (summarize long chunks)
