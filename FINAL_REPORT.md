# Final Report - Quant PDF RAG Retrieval Project

## Project Goal

Improve a Python PDF RAG system for finance / factor-investing papers, with special focus on:

- `quant1.pdf`
- `quant2.pdf`
- `quant3.pdf`

The work started with unstable section detection and ended with a usable multi-PDF retrieval pipeline that can be validated on real factor-investing questions.

---

## Scope of Work Completed

### 1. Section detection improvements

Core files worked on:

- `app/processing/text_splitter.py`
- `app/ingestion/pdf_loader.py`
- `debug_chunks.py`

Main upgrades:

- moved from weak text-only heuristics toward page-level, header-driven section detection
- preserved short header-like blocks instead of dropping them too aggressively
- added layout metadata to PDF parsing:
  - `lines_meta`
  - `body_font_size`
- improved debugging by making `debug_chunks.py` accept a target PDF path

### 2. Retrieval pipeline validation and repair

Core files worked on:

- `app/vector_store/chroma_store.py`
- `app/retrieval/retriever.py`
- `app/retrieval/query_router.py`
- `app/retrieval/query_expansion.py`
- `app/main.py`
- evaluation scripts under `evaluation/`

Main upgrades:

- fixed ChromaDB collection lifecycle issues after reset/recreate
- removed stale `sample.pdf` dependency from the main testing path
- switched evaluation flow to quant-file-based validation
- tuned retrieval routing for weak quant3 query types

### 3. End-to-end validation

Validated not only retrieval hit rate, but also final answer generation using Kimi with retrieved context.

---

## Final Code / Repo State

Important commits in the final phase:

- `e8bc3bd` - Preserve PDF layout metadata for section debugging
- `14ea638` - Fix Chroma reset lifecycle for quant evaluation
- `0179498` - Tune retrieval routing for quant3 performance and robustness

`sample.pdf` was removed from `data/raw/` because it was no longer part of the active quant-paper workflow.

---

## Section Detection Results

These are practical estimates for **main section accuracy**, not perfect page-by-page gold scoring.

- `quant1`: ~85-90%
- `quant2`: ~75-85% (near-pass / usable)
- `quant3`: ~65-75% before retrieval-side compensation, but good enough to support the next stage

Interpretation:

- `quant1` became broadly usable
- `quant2` reached a reasonable main-structure checkpoint
- `quant3` remained the weakest structurally, but retrieval-side improvements compensated for this enough to move forward

---

## Retrieval Evaluation Method

The retrieval benchmark uses `evaluation/evaluator.py`.

It does **not** score only final answer correctness.
It first evaluates whether retrieved chunks hit the expected section(s) for each question.

Main retrieval metrics:

- `important_section_hit_rate`
- `avg_score`
- per-query-type breakdown

`avg_score` combines:

- important hit
- relevance keyword overlap
- noise penalty
- structure alignment

---

## Retrieval Results (Post-Fix)

### Quant1

- average score: **0.90**
- important section hit rate: **1.00**
- hits: **9 / 9**

### Quant2

- average score: **0.91**
- important section hit rate: **1.00**
- hits: **9 / 9**

### Quant3

Before targeted retrieval tuning:

- average score: **0.76**
- important section hit rate: **0.78**
- hits: **7 / 9**

After targeted retrieval tuning for `performance` and `robustness`:

- average score: **0.87**
- important section hit rate: **1.00**
- hits: **9 / 9**

Interpretation:

- quant1 is stable
- quant2 is stable
- quant3 was the main weak point, but retrieval-side tuning closed the gap

---

## End-to-End Answer Validation

End-to-end validation means:

1. ingest PDF
2. retrieve chunks
3. generate final answer with Kimi
4. judge whether the answer is grounded and useful

### Quant1

Strengths:

- contribution-style answers are grounded
- performance / robustness questions are handled reasonably

Weakness:

- limitations questions tend to be answered conservatively (`I could not find the answer...`)

### Quant2

Strengths:

- benchmark and performance questions are strong
- final answers are often specific and grounded

Weakness:

- robustness questions remain weaker end-to-end than they look at retrieval-only level
- limitations answers are usable but somewhat brittle

### Quant3

Strengths:

- contribution question is strong
- performance question improved significantly after routing tweaks
- limitations answers became reasonable

Weakness:

- robustness still remains the weakest end-to-end category
- the system often prefers a conservative "not found" answer rather than stretching beyond the context

---

## Final Assessment

### What is clearly working

- multi-PDF ingestion for quant papers
- layout-aware section metadata extraction
- quant-aware retrieval routing
- ChromaDB reset / rebuild workflow
- retrieval evaluation on `quant1/quant2/quant3`
- end-to-end QA for many important query types

### What is good enough to ship as a project checkpoint

- section detection is usable enough to support downstream retrieval
- retrieval quality is strong across all three target PDFs
- quant3 was materially improved instead of left as a weak outlier

### What is still imperfect

- section labels are still not "perfect gold-standard structure"
- robustness-type final answers are weaker than contribution / benchmark / performance answers
- limitations answers are sometimes cautious and under-extractive

---

## Recommended Next Steps

Two sensible paths remain:

### Option A - Project packaging / portfolio polish

- tighten README wording
- add example queries and outputs
- summarize benchmark and end-to-end results clearly
- present known weaknesses honestly

### Option B - Final end-to-end robustness pass

Focus only on the weakest remaining category:

- `robustness`
- possibly `limitations`

Likely files:

- `app/retrieval/query_expansion.py`
- `app/retrieval/query_router.py`
- `app/llm/kimi_client.py` or answer prompting logic

---

## Bottom Line

This project successfully moved from:

- unstable section detection
- single-paper assumptions
- broken quant-file evaluation flow

to:

- quant-file-based validation
- stable retrieval on quant1 / quant2 / quant3
- usable end-to-end QA for the most important finance-paper question types

The strongest final claim that is supported by evidence is:

> The system is now a usable, quant-paper-oriented PDF RAG pipeline with strong retrieval performance across three target papers, plus mostly grounded end-to-end answers, with robustness-style answering remaining the main known weakness.
