# RAG Retrieval Improvement Project

## Project Overview
Self-improving retrieval pipeline for a PDF-based RAG chatbot. The system autonomously evaluates and improves retrieval quality through iterative experimentation.

## Current Pipeline
```
PDF → Text Extraction → Semantic Chunking → Embeddings → ChromaDB → Section-Boosted Retrieval → Kimi LLM
```

## Results Summary

| Iteration | Score | Hit Rate | Key Improvement |
|-----------|-------|----------|-----------------|
| Baseline | 0.41 | 20% | - |
| **Iter 1**: Semantic Chunking | 0.73 | 90% | Sentence-aware + section detection |
| **Iter 2**: Fix Acknowledgments | 0.79 | 100% | Filter acknowledgments + query expansion |
| **Iter 3**: Cross-Encoder Rerank | **0.85** | 90% | Two-stage retrieval (embedding + cross-encoder) |

**Total Improvement**: 0.41 → 0.85 (+107%)

## Benchmark
10 questions covering key paper aspects:
- Abstract content retrieval
- Problem statement identification
- Method comparison (FINN vs Monte Carlo)
- Technical details (PDE, Greeks, accuracy)

## Running Evaluations

```bash
# Baseline (original pipeline)
python evaluation/run_baseline.py

# Iteration 1 (current)
python evaluation/run_iteration_1.py
```

## Project Structure
```
rag_project/
├── app/
│   ├── ingestion/        # PDF loading
│   ├── processing/       # Semantic chunking
│   ├── embeddings/       # BGE embeddings
│   ├── vector_store/     # ChromaDB with section metadata
│   └── retrieval/        # Section-boosted retriever
├── evaluation/
│   ├── benchmark/        # Test questions
│   ├── evaluator.py      # Scoring framework
│   └── experiments/      # Iteration logs
└── data/raw/             # PDF documents
```

## Next Iterations
Potential improvements to explore:
1. Query expansion for technical terms (address Q4 miss)
2. Hybrid retrieval (sparse + dense)
3. Reranking with cross-encoder
4. Query rewriting for better semantic match
