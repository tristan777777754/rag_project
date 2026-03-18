# Retrieval Improvement Experiment Log

## Iteration 1: Semantic Chunking with Section Detection

### Date
2026-03-18

### Diagnosis
The baseline pipeline had severe retrieval quality issues:
- Only 20% of questions retrieved the expected page (usually page 1/abstract)
- Fixed 800-character chunks cut mid-word (e.g., "ntiation" from "differentiation")
- No semantic boundaries - sentences split arbitrarily
- Page 1 content (abstract) not prioritized despite being most important
- Retrieved pages were random (29, 37, 26) instead of relevant sections

### Proposed Change
Implement semantic chunking with section detection:
1. Split on sentence boundaries instead of fixed character counts
2. Detect document sections (abstract, introduction, methodology, references)
3. Filter out noisy chunks (references, tiny fragments)
4. Boost abstract/introduction sections during retrieval

### Files Modified
- `app/processing/text_splitter.py` - Complete rewrite with semantic chunking
- `app/vector_store/chroma_store.py` - Added section metadata storage and section boosting
- `app/retrieval/retriever.py` - Updated to use section-boosted search

### Benchmark Results

#### Before (Baseline)
- Average Score: 0.41 / 1.0
- Important Section Hit Rate: 20%
- Hits: 2/10 | Misses: 8/10

#### After (Iteration 1)
- Average Score: 0.73 / 1.0
- Important Section Hit Rate: 90%
- Hits: 9/10 | Misses: 1/10

### Key Wins
1. **Q3 (Summarize abstract)**: Now retrieves page 1 three times - direct abstract access
2. **Q5 (FINN vs Monte Carlo)**: Page 1 retrieved with specific accuracy numbers
3. **Q8 (FINN accuracy)**: Page 1 retrieved with "0.04 to 0.07 cents" metric
4. **Q10 (Auto differentiation)**: Page 1 retrieved with explanation

### Remaining Issue
- **Q4 (PDE role)**: Still missing page 1 - may need query expansion or better PDE keyword handling

### Decision
✅ **KEEP** - Significant improvement across all metrics. The semantic chunking approach successfully captures document structure and prioritizes important sections.

### Notes
- Section detection is heuristic-based and works well for academic papers
- Abstract section correctly identified on page 1 (5 chunks)
- Reference filtering removed 11 noisy chunks
- Section boosting gives abstract 1.5x priority, introduction 1.3x

### Next Steps
1. Address Q4 miss (PDE role question)
2. Consider query expansion for technical terms
3. Evaluate if additional improvements needed or if current quality is sufficient
