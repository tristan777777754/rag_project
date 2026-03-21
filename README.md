# Quant PDF RAG Project

A PDF RAG system for finance and factor-investing papers, tuned on three target documents:

- `quant1.pdf`
- `quant2.pdf`
- `quant3.pdf`

This repo evolved from a generic PDF chatbot into a more domain-aware retrieval pipeline with:

- layout-aware section detection
- semantic chunking
- hybrid retrieval
- query-aware routing
- quant-paper validation
- end-to-end answer testing

---

## What This Project Does

The system ingests academic PDFs, splits them into semantically meaningful chunks, stores them in ChromaDB, retrieves relevant chunks with dense + sparse search, and answers user questions with Kimi using only retrieved context.

Pipeline:

```text
PDF
  -> PDF loader with layout metadata
  -> section-aware semantic chunking
  -> embeddings + ChromaDB
  -> BM25 + dense retrieval
  -> query-aware routing / boosting
  -> top chunks to Kimi
  -> grounded answer
```

---

## Target Papers

Main validation in this phase was done on:

- `data/raw/quant1.pdf`
- `data/raw/quant2.pdf`
- `data/raw/quant3.pdf`

These are finance / factor-investing papers with different formatting styles, which made section detection and retrieval generalization the main challenge.

---

## Main Technical Pieces

### 1. PDF loading with lightweight layout metadata

File:
- `app/ingestion/pdf_loader.py`

The loader now preserves more than plain text. It also keeps:

- `lines_meta`
- `body_font_size`

This supports layout-aware section detection instead of relying only on raw text.

### 2. Section-aware semantic chunking

File:
- `app/processing/text_splitter.py`

The chunker moved beyond fixed-size text splitting and now uses:

- sentence-aware chunking
- page-level section inference
- header-driven logic
- finance-paper-oriented broad-section mapping

Main canonical sections used downstream:

- `front_matter`
- `abstract`
- `introduction`
- `methodology`
- `results`
- `conclusion`
- `references`
- `body`

### 3. Hybrid retrieval

Files:
- `app/retrieval/retriever.py`
- `app/retrieval/bm25_retriever.py`
- `app/embeddings/embedder.py`
- `app/vector_store/chroma_store.py`

The retriever combines:

- BM25 keyword retrieval
- dense embedding retrieval
- weighted hybrid merging
- section-aware reranking

### 4. Query-aware routing for quant papers

Files:
- `app/retrieval/query_router.py`
- `app/retrieval/query_expansion.py`

Supported query intents include:

- contribution
- factor definition
- data sample
- methodology / portfolio construction
- benchmark
- performance
- robustness
- limitations

This routing layer became especially important for `quant3`, where answers are often spread across abstract, conclusion, and discussion-style pages rather than a clean `results` section.

### 5. End-to-end answer generation

Files:
- `app/llm/kimi_client.py`
- `app/main.py`

Kimi is prompted to answer strictly from retrieved context.

---

## Validation Summary

### A. Section detection checkpoint

These are practical estimates for **main section accuracy**:

- `quant1`: ~85-90%
- `quant2`: ~75-85%
- `quant3`: ~65-75% before retrieval-side compensation

Interpretation:

- `quant1` is broadly usable
- `quant2` is near-pass / usable
- `quant3` is structurally weaker, but still workable for downstream retrieval after routing improvements

### B. Retrieval evaluation

Evaluation is driven by:

- `evaluation/evaluator.py`
- `evaluation/benchmark/questions.json`

The benchmark measures retrieval quality using expected section hits, relevance heuristics, noise, and structure alignment.

Final retrieval results:

- `quant1`: avg_score **0.90**, hit_rate **1.00**
- `quant2`: avg_score **0.91**, hit_rate **1.00**
- `quant3`: avg_score **0.87**, hit_rate **1.00**

Note: `quant3` originally underperformed and was improved with targeted routing changes for:

- `performance`
- `robustness`

### C. End-to-end answer validation

We also tested the full path:

- retrieve chunks
- send context to Kimi
- inspect final answer quality

Overall result:

- contribution / benchmark / performance questions are mostly solid
- limitations questions are mixed but usable
- robustness remains the weakest end-to-end question type

---

## Important Fixes Made During This Phase

### Layout-aware debugging support

- preserved short header-like blocks
- added `lines_meta` and `body_font_size`
- made `debug_chunks.py` accept a target PDF path

### Chroma reset lifecycle fix

A real blocker appeared during quant-file evaluation:

- collection reset worked
- but retrieval still held a stale collection handle
- which caused `Collection does not exist` errors

This was fixed by making collection access go through a fresh getter instead of relying on a stale imported object.

### Removed old `sample.pdf` workflow

The active project workflow is now quant-paper-based.

- `data/raw/sample.pdf` was removed
- evaluation scripts were updated away from the old sample-file assumption

---

## Key Files

```text
rag_project/
├── app/
│   ├── ingestion/
│   │   └── pdf_loader.py
│   ├── processing/
│   │   └── text_splitter.py
│   ├── embeddings/
│   │   └── embedder.py
│   ├── vector_store/
│   │   └── chroma_store.py
│   ├── retrieval/
│   │   ├── retriever.py
│   │   ├── bm25_retriever.py
│   │   ├── query_router.py
│   │   ├── query_expansion.py
│   │   └── reranker.py
│   ├── llm/
│   │   └── kimi_client.py
│   └── main.py
├── evaluation/
│   ├── benchmark/
│   │   └── questions.json
│   ├── evaluator.py
│   └── run_*.py
├── data/
│   └── raw/
│       ├── quant1.pdf
│       ├── quant2.pdf
│       ├── quant3.pdf
│       └── sample2.pdf
├── debug_chunks.py
├── FINAL_REPORT.md
└── requirements.txt
```

---

## Running the Project

### Install

```bash
pip install -r requirements.txt
```

### Set API key

```bash
echo "MOONSHOT_API_KEY=your_key_here" > .env
```

### Run the app

```bash
python -m app.main
```

### Debug section mapping for a target PDF

```bash
python3 debug_chunks.py data/raw/quant1.pdf
python3 debug_chunks.py data/raw/quant2.pdf
python3 debug_chunks.py data/raw/quant3.pdf
```

### Run retrieval evaluation

Existing evaluation scripts are still available under `evaluation/`, but the most reliable current workflow is to ingest one quant PDF at a time and evaluate against `evaluation/benchmark/questions.json`.

---

## Current Strengths

- retrieval is strong across all three quant papers
- the system is no longer tied to a single legacy sample PDF
- query-aware routing helps compensate for imperfect document structure
- end-to-end answers are often grounded and useful for core finance-paper questions

## Current Weaknesses

- section detection is still not perfect gold-standard structure parsing
- robustness-style questions are the weakest end-to-end category
- limitations-style answers can be conservative or under-extractive
- text-only pipeline still ignores tables, figures, and formulas

---

## Recommended Next Steps

If continuing from here, the most valuable next move is **not** more section-detector tuning.

Better options are:

1. improve end-to-end answering for robustness / limitations queries
2. package the repo as a portfolio/demo project
3. add reproducible evaluation scripts for all three quant PDFs
4. improve table / figure extraction for finance papers

---

## Final Status

This repo is now in a good checkpoint state:

- quant-paper ingestion works
- retrieval evaluation works
- end-to-end QA works for the main question types
- known weaknesses are narrow and identifiable

For the full project summary, see:

- `FINAL_REPORT.md`
