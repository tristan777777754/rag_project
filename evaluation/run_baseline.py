"""
Baseline Evaluation Script
Runs the current retrieval pipeline and evaluates performance.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion.pdf_loader import load_pdf
from app.processing.text_splitter import split_pages_into_chunks
from app.embeddings.embedder import embed_texts, embed_query
from app.vector_store.chroma_store import store_chunks, search_chunks, get_collection_count
from app.retrieval.retriever import retrieve_relevant_chunks
from evaluation.evaluator import RetrievalEvaluator, format_report
import json


def setup_and_evaluate():
    """Setup the vector store and run baseline evaluation."""
    print("=" * 60)
    print("BASELINE EVALUATION - Current Pipeline")
    print("=" * 60)
    print()
    
    # Step 1: Ingest PDF
    print("[1] Ingesting PDF...")
    pages = load_pdf("data/raw/sample.pdf")
    print(f"    Loaded {len(pages)} pages")
    
    chunks = split_pages_into_chunks(pages)
    print(f"    Split into {len(chunks)} chunks")
    
    # Show sample chunks
    print(f"\n    Sample chunk structure:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"    Chunk {i}: Page {chunk['page']}, Length: {len(chunk['text'])} chars")
        print(f"      Preview: {chunk['text'][:80]}...")
    print()
    
    # Step 2: Store embeddings
    print("[2] Storing embeddings...")
    text_list = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(text_list)
    store_chunks(chunks, embeddings)
    print(f"    Stored {get_collection_count()} chunks in vector store")
    print()
    
    # Step 3: Run evaluation
    print("[3] Running benchmark evaluation...")
    print()
    
    evaluator = RetrievalEvaluator("evaluation/benchmark/questions.json")
    
    def retrieval_fn(query: str):
        return retrieve_relevant_chunks(query, top_k=5)
    
    results = evaluator.evaluate_all(retrieval_fn)
    
    # Print report
    report = format_report(results)
    print(report)
    print()
    
    # Save results
    output_path = "evaluation/experiments/baseline_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")
    print()
    
    # Detailed analysis for improvement planning
    print("=" * 60)
    print("ANALYSIS FOR IMPROVEMENT PLANNING")
    print("=" * 60)
    print()
    
    # Analyze misses
    for r in results['results']:
        if not r['important_section_hit']:
            print(f"MISS: Q{r['question_id']} (Expected page {r['expected_page']})")
            print(f"      Retrieved pages: {r['pages_retrieved']}")
            print()
    
    return results


if __name__ == "__main__":
    results = setup_and_evaluate()
