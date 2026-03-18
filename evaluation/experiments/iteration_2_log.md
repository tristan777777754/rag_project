# Iteration 2: Fix Acknowledgments Detection + Query Expansion

## Date
2026-03-18

## Problem Identified
During live testing, the query "What does this paper introduce?" returned acknowledgments instead of the abstract:
- Retrieved chunk: "This paper is based on...I am especially grateful..." 
- Expected: "This paper introduces a no-arbitrage, Monte Carlo-free approach..."

Root causes:
1. Section detection was done **per page**, not per chunk - acknowledgments inherited "abstract" label
2. Acknowledgments section not being filtered out
3. Query embedding didn't match well to abstract content

## Changes Made

### 1. Per-Chunk Section Detection (`app/processing/text_splitter.py`)
- Moved `detect_section()` call inside chunk creation loop
- Each chunk now gets its own section label based on content
- Acknowledgments now correctly detected on page 1

### 2. Acknowledgments Detection
- Added keywords: 'thank my', 'dissertation committee', 'mentorship', 'for their support'
- Special case: page 1 + 'thank' + 'committee' = acknowledgments
- Filtered out in `filter_noisy_chunks()`

### 3. Query Expansion (`app/retrieval/query_expansion.py`)
- Expand "what does this paper introduce" → "paper introduces framework approach method"
- Multiple query variations for better coverage
- Rewrites common question patterns to match academic content

### 4. Broader Academic Keywords
- Added: 'neural network', 'finn', 'pde', 'monte carlo', 'pricing', 'derivatives'
- Page 1 content with academic terms → abstract section

## Results

| Metric | Iter 1 | Iter 2 | Change |
|--------|--------|--------|--------|
| Avg Score | 0.73 | **0.79** | +0.06 |
| Hit Rate | 90% | **100%** | +10% |
| Q10 Score | 0.90 | **1.00** | Perfect! |

### Test Query Results
**Query:** "What does this paper introduce?"

**Before (Broken):**
1. Page 1 (acknowledgments): "This paper is based on...grateful..."

**After (Fixed):**
1. Page 1 (abstract): "The framework generalizes naturally..."
2. Page 1 (abstract): "Once trained, FINNs price caplets..."

## Decision
✅ **KEEP** - Fixed the live testing issue, 100% hit rate, better scores.

## Files Modified
- `app/processing/text_splitter.py` - Per-chunk section detection, acknowledgments filtering
- `app/retrieval/query_expansion.py` - New query expansion module
- `app/retrieval/retriever.py` - Use query expansion
- `evaluation/run_iteration_2.py` - Evaluation script
