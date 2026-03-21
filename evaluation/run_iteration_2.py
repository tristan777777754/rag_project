"""
Iteration 2: Fix Acknowledgments Detection + Query Expansion
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


def run_iteration_2():
    """Run Iteration 2: Fix acknowledgments detection and add query expansion."""
    print("=" * 70)
    print("ITERATION 2: Fix Acknowledgments + Query Expansion")
    print("=" * 70)
    print()
    
    # Reset vector store for clean evaluation
    print("[1] Resetting vector store...")
    reset_collection()
    print("    ✓ Vector store reset")
    print()
    
    # Step 1: Ingest PDF
    print("[2] Ingesting PDF with improved chunking...")
    pages = load_pdf("data/raw/quant1.pdf")
    print(f"    Loaded {len(pages)} pages")
    
    # Use new semantic chunking
    chunks = split_pages_into_chunks_semantic(pages, max_chunk_size=800, overlap_sentences=1)
    print(f"    Created {len(chunks)} semantic chunks")
    
    # Show page 1 chunks with section info
    print(f"\n    Page 1 chunk breakdown:")
    for i, c in enumerate(chunks):
        if c['page'] == 1:
            print(f"    Chunk {i}: Section={c['section']}, Length={len(c['text'])} chars")
            preview = c['text'][:60].replace('\n', ' ')
            print(f"             Preview: {preview}...")
    print()
    
    # Filter noisy chunks
    filtered_chunks = filter_noisy_chunks(chunks)
    removed = len(chunks) - len(filtered_chunks)
    print(f"    Filtered out {removed} noisy chunks, {len(filtered_chunks)} remaining")
    
    # Show section distribution
    section_counts = {}
    for chunk in filtered_chunks:
        sec = chunk['section']
        section_counts[sec] = section_counts.get(sec, 0) + 1
    print("    Section distribution:")
    for sec, count in sorted(section_counts.items()):
        print(f"      {sec}: {count}")
    print()
    
    # Step 2: Store embeddings
    print("[3] Storing embeddings...")
    text_list = [chunk["text"] for chunk in filtered_chunks]
    embeddings = embed_texts(text_list)
    store_chunks(filtered_chunks, embeddings)
    print(f"    Stored {get_collection_count()} chunks in vector store")
    print()
    
    # Step 3: Test specific query
    print("[4] Testing specific query...")
    query = "What does this paper introduce?"
    print(f"    Query: '{query}'")
    test_chunks = retrieve_relevant_chunks(query, top_k=5, use_section_boost=True)
    print("\n    Retrieved chunks:")
    for i, c in enumerate(test_chunks):
        print(f"    {i+1}. Page {c['page']} ({c['section']}): {c['text'][:80]}...")
    print()
    
    # Step 5: Run full evaluation
    print("[5] Running benchmark evaluation...")
    print()
    
    evaluator = RetrievalEvaluator("evaluation/benchmark/questions.json")
    
    def retrieval_fn(query: str):
        return retrieve_relevant_chunks(query, top_k=5, use_section_boost=True)
    
    results = evaluator.evaluate_all(retrieval_fn)
    
    # Print report
    report = format_report(results)
    print(report)
    print()
    
    # Save results
    output_path = "evaluation/experiments/iteration_2_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")
    print()
    
    # Compare improvements
    print("=" * 70)
    print("COMPARISON: Iteration 1 vs Iteration 2")
    print("=" * 70)
    print()
    
    # Load iteration 1
    with open("evaluation/experiments/iteration_1_results.json", 'r') as f:
        prev = json.load(f)
    
    prev_score = prev['summary']['avg_score']
    prev_hit_rate = prev['summary']['important_section_hit_rate']
    
    new_score = results['summary']['avg_score']
    new_hit_rate = results['summary']['important_section_hit_rate']
    
    print(f"Metric                | Iter 1   | Iter 2      | Change")
    print(f"----------------------|----------|-------------|--------")
    print(f"Avg Score             | {prev_score:8.2f} | {new_score:11.2f} | {'+' if new_score > prev_score else ''}{new_score - prev_score:.2f}")
    print(f"Important Section Hit | {prev_hit_rate:7.0%}   | {new_hit_rate:10.0%}    | {'+' if new_hit_rate > prev_hit_rate else ''}{new_hit_rate - prev_hit_rate:.0%}")
    print()
    
    # Decision
    if new_score >= prev_score and new_hit_rate >= prev_hit_rate:
        print("✓ DECISION: KEEP - Maintained or improved performance!")
    elif new_hit_rate > prev_hit_rate:
        print("✓ DECISION: KEEP - Hit rate improved")
    else:
        print("⚠ DECISION: MIXED - Review specific changes")
    
    return results


if __name__ == "__main__":
    results = run_iteration_2()
