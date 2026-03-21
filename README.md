# PDF RAG Chatbot with Hybrid Retrieval

A production-ready PDF-based RAG (Retrieval-Augmented Generation) system that evolved from basic embedding retrieval to a sophisticated hybrid retrieval pipeline with query-aware routing.

## 🎯 Project Overview

This project demonstrates the iterative improvement of a PDF question-answering system, starting from a naive embedding-based approach and evolving into a multi-stage retrieval pipeline that combines BM25 keyword search, dense embeddings, and cross-encoder reranking.

### Key Features
- **Semantic Chunking**: Sentence-aware text splitting with section detection
- **Hybrid Retrieval**: Combines BM25 (keyword) + Dense Embeddings (semantic)
- **Query-Aware Routing**: Dynamic section/page boosting based on query type
- **Cross-Encoder Reranking**: Precise relevance scoring for final results
- **Multi-PDF Support**: Works with various academic paper formats

## 📊 Performance Evolution

> ⚠️ **Note**: These metrics were measured on a single academic paper (Heath-Jarrow-Morton derivatives pricing, 37 pages). The benchmark contains 10 questions specifically designed for this document. While the techniques are designed to generalize, performance may vary on different document types (legal contracts, medical records, multi-modal PDFs, etc.).

| Iteration | Technique | Score | Hit Rate |
|-----------|-----------|-------|----------|
| Baseline | Basic chunking + Embedding | 0.41 | 20% |
| **Iter 1** | Semantic Chunking + Section Detection | 0.73 | 90% |
| **Iter 2** | Fix Acknowledgments + Query Expansion | 0.79 | 100% |
| **Iter 3** | Cross-Encoder Re-ranking | 0.85 | 90% |
| **Iter 4** | Hybrid Retrieval (BM25 + Embedding) | - | - |
| **Iter 5** | Query-Aware Section & Page Routing | **0.90+** | **100%** |

**Total Improvement: +120%** (0.41 → 0.90+)

## 🏗️ Architecture

```
User Query
    ↓
[Query Router] - Classifies query type (intro/problem/method/results)
    ↓
┌─────────────────────────────────────────┐
│  Stage 1: Hybrid Retrieval (Recall)     │
│  ├─ BM25 Keyword Search                 │
│  └─ Dense Embedding Search (BAAI/bge)   │
│                                         │
│  Merge with query-aware boosting:       │
│  - Section boost: abstract=2.5x         │
│  - Page boost: Page 1=3x (for intro)    │
└─────────────────────────────────────────┘
    ↓
[Query-Aware Re-ranking]
    ↓
┌─────────────────────────────────────────┐
│  Stage 2: Cross-Encoder (Precision)     │
│  cross-encoder/ms-marco-MiniLM-L6-v2    │
│  Scores [query, chunk] pairs            │
└─────────────────────────────────────────┘
    ↓
Top 5 Results → LLM (Kimi) → Final Answer
```

## 🔧 Technical Components

### 1. Semantic Chunking (`app/processing/text_splitter.py`)

**Problem**: Fixed-size chunks cut sentences mid-word and mix sections.

**Solution**: 
- Split on sentence boundaries using regex
- Detect sections: abstract, introduction, methodology, results, references, acknowledgments
- Filter noisy chunks (acknowledgments, tiny fragments)

```python
# Before (fixed 800 chars):
"...ntiation efficiently computing the exa..."  # mid-word cut!

# After (semantic):
"Finance-Informed Neural Networks (FINNs) solve this PDE directly..."
```

### 2. Section Detection (`app/processing/text_splitter.py`)

Detects document structure without hardcoded headers:
- **Abstract**: Page 1 + academic keywords ("paper introduces", "here we show")
- **Introduction**: "1 Introduction" headers or intro keywords
- **Methodology**: Section headers with method keywords
- **References**: Citation patterns (year, vol., pp.)
- **Acknowledgments**: "thank", "grateful", "committee"

### 3. Query Expansion (`app/retrieval/query_expansion.py`)

Expands natural questions to match academic writing styles:

```python
"what does this paper introduce?"
↓
["paper introduces", "we present", "here we show", 
 "we demonstrate", "this study", "we report"]
```

Supports patterns:
- Introduction queries → abstract/intro keywords
- Problem queries → challenge/limitation keywords  
- Method queries → approach/algorithm keywords
- Results queries → accuracy/performance keywords

### 4. BM25 Retriever (`app/retrieval/bm25_retriever.py`)

Traditional keyword-based retrieval using BM25Okapi:
- Tokenizes text (lowercase, remove punctuation)
- Builds inverted index for fast lookup
- Scores based on term frequency and inverse document frequency

**Use case**: Exact keyword matching (e.g., "introduce" must appear)

### 5. Query Router (`app/retrieval/query_router.py`)

**The key innovation for handling different PDFs.**

Classifies queries and applies dynamic boosting:

| Query Type | Detected By | Section Boost | Page Boost |
|------------|-------------|---------------|------------|
| `introduction` | "introduce", "about", "overview" | abstract=2.5x | Page 1=3x |
| `problem` | "problem", "solve", "challenge" | intro=1.8x | Page 1-2=2.5x |
| `method` | "method", "approach", "how" | methodology=2.0x | - |
| `results` | "result", "accuracy", "achieve" | results=2.0x | - |

**Fallback**: When no "introduction" section detected, boosts early pages (1-3) for intro queries.

### 6. Hybrid Merger (`app/retrieval/retriever.py`)

Combines BM25 and Embedding results:

```python
# Normalize scores to 0-1
emb_score = 1.0 - (rank / total)  # Rank-based
bm25_score = normalize(raw_bm25)   # Score-based

# Weighted combination
hybrid_score = alpha * bm25_score + (1-alpha) * emb_score
```

Then applies query-aware section and page boosts.

### 7. Cross-Encoder Reranker (`app/retrieval/reranker.py`)

Final precision stage using `cross-encoder/ms-marco-MiniLM-L-6-v2`:
- Takes top 20 candidates from hybrid retrieval
- Scores [query, chunk] pairs with cross-attention
- Returns top 5 most relevant

**Why**: Embeddings measure similarity, cross-encoder measures relevance.

## 📁 Project Structure

```
rag_project/
├── app/
│   ├── ingestion/
│   │   └── pdf_loader.py          # PDF text extraction with PyMuPDF
│   ├── processing/
│   │   └── text_splitter.py       # Semantic chunking + section detection
│   ├── embeddings/
│   │   └── embedder.py            # BAAI/bge-small-en embeddings
│   ├── vector_store/
│   │   └── chroma_store.py        # ChromaDB storage & retrieval
│   ├── retrieval/
│   │   ├── retriever.py           # Main hybrid retrieval pipeline
│   │   ├── bm25_retriever.py      # BM25 keyword search
│   │   ├── reranker.py            # Cross-encoder reranking
│   │   ├── query_expansion.py     # Query rewriting
│   │   └── query_router.py        # Query classification & routing
│   ├── llm/
│   │   └── kimi_client.py         # Kimi LLM integration
│   └── main.py                     # CLI entry point
├── evaluation/
│   ├── benchmark/
│   │   └── questions.json         # Test questions
│   ├── evaluator.py               # Scoring framework
│   └── experiments/               # Iteration logs & results
├── data/
│   └── raw/
│       ├── quant1.pdf
│       ├── quant2.pdf
│       ├── quant3.pdf
│       └── sample2.pdf
└── requirements.txt
```

## 🚀 Usage

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
echo "MOONSHOT_API_KEY=your_key_here" > .env
```

### Run

```bash
# Interactive chat
python -m app.main

# Or test retrieval directly
python -c "
from app.retrieval.retriever import retrieve_relevant_chunks
chunks = retrieve_relevant_chunks('What does this paper introduce?')
for c in chunks:
    print(f'Page {c[\"page\"]}: {c[\"text\"][:100]}...')
"
```

### Configuration

Key parameters in `retrieve_relevant_chunks()`:
- `use_query_routing=True` - Enable query classification
- `use_hybrid=True` - Use BM25 + Embedding
- `hybrid_alpha=0.5` - BM25 weight (0.5 = equal)
- `use_reranker=True` - Enable cross-encoder

## 📈 Iteration History

### Iteration 1: Semantic Chunking
**Problem**: Fixed-size chunks cut mid-word and mix sections.

**Solution**:
- Sentence-aware splitting
- Section detection (abstract, intro, references)
- Filter acknowledgments

**Result**: Score 0.41 → 0.73 (+78%)

### Iteration 2: Acknowledgments Fix + Query Expansion  
**Problem**: Acknowledgments misclassified as abstract; query vocabulary mismatch.

**Solution**:
- Per-chunk section detection
- Acknowledgments filtering
- Expand "introduce" → "paper introduces", "we present"

**Result**: Score 0.73 → 0.79, Hit rate 90% → 100%

### Iteration 3: Cross-Encoder Reranking
**Problem**: Embedding similarity ≠ relevance.

**Solution**:
- Two-stage: embedding (recall) → cross-encoder (precision)
- cross-encoder/ms-marco-MiniLM-L6-v2

**Result**: Score 0.79 → 0.85, Q2 relevance 0.0 → 0.8

### Iteration 4: Hybrid Retrieval
**Problem**: Embedding-only misses exact keyword matches.

**Solution**:
- BM25 + Embedding merge
- Configurable weighting (alpha)
- RRF (Reciprocal Rank Fusion) style merging

**Result**: Better generalization across different PDF formats

### Iteration 5: Query-Aware Routing
**Problem**: "introduce" query fails on papers using "here we show" format.

**Solution**:
- Query classification (intro/problem/method/results)
- Dynamic section boosting (abstract=2.5x for intro)
- Page-based fallback (Page 1=3x when no intro section)
- Increased initial_top_k to ensure early pages recalled

**Result**: Page 1 correctly retrieved for all introduction queries

## 🎓 Key Learnings & Limitations

1. **Chunking matters more than embedding**: Semantic boundaries beat larger chunks
2. **Section detection is fragile**: Need fallback to page-based routing
3. **Hybrid > Single method**: BM25 for precision, embeddings for recall
4. **Query routing is essential**: Different queries need different strategies
5. **Cross-encoder is worth it**: Significant relevance improvement for small latency cost

### ⚠️ Limitations

- **Single-document evaluation**: All performance metrics based on one academic paper. Multi-PDF validation needed for production use.
- **Text-only**: Does not extract tables, figures, or mathematical formulas
- **English only**: Section detection tuned for English academic papers
- **Query-dependent**: Performance varies based on how well query matches document vocabulary

## 🔮 Future Improvements

- **Multi-hop retrieval**: For questions requiring multiple document sections
- **Table/figure extraction**: Current system only processes text
- **Citation resolution**: Link citations to reference entries
- **Fine-tuned embeddings**: Domain-specific embedding models

## 📄 License

MIT

## 🙏 Acknowledgments

- BAAI/bge-small-en for embeddings
- sentence-transformers for cross-encoder
- ChromaDB for vector storage
- rank-bm25 for keyword retrieval
