"""
Iteration 3 Evaluation: Cross-Encoder Re-ranking
Tests if reranking improves content relevance.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion.pdf_loader import load_pdf
from app.processing.text_splitter import split_pages_into_chunks_semantic, filter_noisy_chunks
from app.embeddings.embedder import embed_texts
from app.vector_store.chroma_store import store_chunks, get_collection_count, reset_collection
from app.retrieval.retriever import retrieve_relevant_chunks
from evaluation.evaluator import RetrievalEvaluator, format_report
import json


def run_iteration_3():
    """Run Iteration 3: Cross-Encoder Re-ranking."""
    print("=" * 70)
    print("ITERATION 3: Cross-Encoder Re-ranking")
    print("=" * 70)
    print()
    
    # Reset vector store for clean evaluation
    print("[1] Resetting vector store...")
    reset_collection()
    print("    ✓ Vector store reset")
    print()
    
    # Step 1: Ingest PDF
    print("[2] Ingesting PDF...")
    pages = load_pdf("data/raw/sample.pdf")
    print(f"    Loaded {len(pages)} pages")
    
    chunks = split_pages_into_chunks_semantic(pages, max_chunk_size=800, overlap_sentences=1)
    filtered_chunks = filter_noisy_chunks(chunks)
    print(f"    Created {len(chunks)} chunks, {len(filtered_chunks)} after filtering")
    
    # Store embeddings
    text_list = [chunk["text"] for chunk in filtered_chunks]
    embeddings = embed_texts(text_list)
    store_chunks(filtered_chunks, embeddings)
    print(f"    Stored {get_collection_count()} chunks")
    print()
    
    # Step 2: Test comparison - with vs without reranking
    print("[3] Testing: Embedding-only vs Cross-Encoder Re-ranking")
    print()
    
    test_query = "What problem does this paper solve?"
    print(f"    Query: '{test_query}'")
    print()
    
    # Without reranking
    print("    --- WITHOUT Reranking ---")
    chunks_no_rerank = retrieve_relevant_chunks(test_query, top_k=5, use_reranker=False)
    for i, c in enumerate(chunks_no_rerank[:3]):
        print(f"    {i+1}. Page {c['page']}: {c['text'][:70]}...")
    print()
    
    # With reranking
    print("    --- WITH Cross-Encoder Reranking ---")
    chunks_with_rerank = retrieve_relevant_chunks(test_query, top_k=5, use_reranker=True)
    for i, c in enumerate(chunks_with_rerank[:3]):
        score = c.get('rerank_score', 'N/A')
        print(f"    {i+1}. Page {c['page']} (score: {score:.2f}): {c['text'][:60]}...")
    print()
    
    # Step 3: Full evaluation WITH reranking
    print("[4] Running full benchmark WITH reranking...")
    print()
    
    evaluator = RetrievalEvaluator("evaluation/benchmark/questions.json")
    
    def retrieval_fn(query: str):
        return retrieve_relevant_chunks(query, top_k=5, use_reranker=True)
    
    results = evaluator.evaluate_all(retrieval_fn)
    
    # Print report
    report = format_report(results)
    print(report)
    print()
    
    # Save results
    output_path = "evaluation/experiments/iteration_3_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")
    print()
    
    # Compare all iterations
    print("=" * 70)
    print("COMPARISON: All Iterations")
    print("=" * 70)
    print()
    
    iterations = [
        ("Baseline", "evaluation/experiments/baseline_results.json"),
        ("Iter 1: Semantic", "evaluation/experiments/iteration_1_results.json"),
        ("Iter 2: Ack Fix", "evaluation/experiments/iteration_2_results.json"),
        ("Iter 3: Rerank", "evaluation/experiments/iteration_3_results.json"),
    ]
    
    print(f"{'Iteration':<20} {'Score':<8} {'Hit Rate':<10} {'Improvement'}")
    print("-" * 55)
    
    baseline_score = None
    for name, path in iterations:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            score = data['summary']['avg_score']
            hit_rate = data['summary']['important_section_hit_rate']
            
            if baseline_score is None:
                baseline_score = score
                imp = "baseline"
            else:
                delta = score - baseline_score
                imp = f"+{delta:.2f}"
            
            print(f"{name:<20} {score:<8.2f} {hit_rate:<10.0%} {imp}")
        except FileNotFoundError:
            print(f"{name:<20} (not run yet)")
    
    print()
    
    # Decision
    new_score = results['summary']['avg_score']
    prev_score = 0.79  # Iteration 2
    
    if new_score > prev_score:
        print(f"✓ DECISION: KEEP - Score improved from {prev_score:.2f} to {new_score:.2f}!")
    elif new_score >= prev_score - 0.05:  # Allow small drop for relevance quality
        print(f"✓ DECISION: KEEP - Score maintained ({new_score:.2f}), but relevance should be better")
    else:
        print(f"⚠ DECISION: REVIEW - Score dropped to {new_score:.2f}")
    
    return results


if __name__ == "__main__":
    results = run_iteration_3()
