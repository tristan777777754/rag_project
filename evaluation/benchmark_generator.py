"""
Benchmark generator for factor investing paper retrieval.

This module creates a domain-specific benchmark scaffold that can then be filled
with paper-specific gold answers.
"""
import json
from typing import List, Dict


FACTOR_BENCHMARK_TEMPLATE = [
    {
        "id": "q1",
        "query": "What factor does this paper propose?",
        "query_type": "contribution",
        "gold_section": ["abstract", "introduction"],
        "gold_answer": "TODO: proposed factor or anomaly",
    },
    {
        "id": "q2",
        "query": "How is the factor constructed?",
        "query_type": "factor_definition",
        "gold_section": ["methodology", "introduction"],
        "gold_answer": "TODO: factor construction details",
    },
    {
        "id": "q3",
        "query": "What dataset is used?",
        "query_type": "data_sample",
        "gold_section": ["methodology", "results"],
        "gold_answer": "TODO: dataset names / asset universe",
    },
    {
        "id": "q4",
        "query": "What is the sample period?",
        "query_type": "data_sample",
        "gold_section": ["methodology", "results"],
        "gold_answer": "TODO: sample period",
    },
    {
        "id": "q5",
        "query": "How are portfolios constructed?",
        "query_type": "methodology_portfolio",
        "gold_section": ["methodology", "results"],
        "gold_answer": "TODO: sorting / weighting / long-short construction",
    },
    {
        "id": "q6",
        "query": "What benchmark is used?",
        "query_type": "benchmark",
        "gold_section": ["results", "methodology"],
        "gold_answer": "TODO: benchmark model",
    },
    {
        "id": "q7",
        "query": "What performance metrics are reported?",
        "query_type": "performance",
        "gold_section": ["results", "abstract"],
        "gold_answer": "TODO: alpha / returns / Sharpe / t-stats",
    },
    {
        "id": "q8",
        "query": "Are transaction costs or robustness checks discussed?",
        "query_type": "robustness",
        "gold_section": ["results", "conclusion"],
        "gold_answer": "TODO: robustness checks",
    },
    {
        "id": "q9",
        "query": "What limitations does the paper mention?",
        "query_type": "limitations",
        "gold_section": ["conclusion", "abstract"],
        "gold_answer": "TODO: limitations or caveats",
    },
]


def generate_benchmark_from_chunks(chunks: List[Dict], num_questions: int = 9) -> List[Dict]:
    """
    Return a factor-investing benchmark scaffold.

    We keep this deterministic and template-based because gold answers should be
    curated manually for each target paper.
    """
    benchmark = FACTOR_BENCHMARK_TEMPLATE[:num_questions]

    available_sections = sorted({str(chunk.get('section', 'body')).lower() for chunk in chunks})
    for item in benchmark:
        item['notes'] = (
            f"Available detected sections in this document: {available_sections}. "
            "Review and replace TODO gold_answer with a paper-specific answer before scoring."
        )

    return benchmark


def save_universal_benchmark(chunks: List[Dict], output_path: str):
    """Generate and save a factor-investing benchmark template."""
    questions = generate_benchmark_from_chunks(chunks)

    with open(output_path, 'w') as f:
        json.dump(questions, f, indent=2)

    print(f"Generated {len(questions)} factor-investing benchmark questions")
    print(f"Saved to: {output_path}")
    for q in questions:
        print(f"  {q['id']}: {q['query']} ({q['query_type']})")

    return questions


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '../..')
    from app.ingestion.pdf_loader import load_pdf
    from app.processing.text_splitter import split_pages_into_chunks_semantic

    pages = load_pdf("data/raw/sample.pdf")
    chunks = split_pages_into_chunks_semantic(pages)

    save_universal_benchmark(chunks, "evaluation/benchmark/auto_generated.json")
