"""
Universal Benchmark Generator
Auto-generates benchmark questions for any PDF using LLM.
"""
import json
from typing import List, Dict


def generate_benchmark_from_chunks(chunks: List[Dict], num_questions: int = 5) -> List[Dict]:
    """
    Auto-generate benchmark questions from document chunks.
    Uses simple heuristics without requiring LLM.
    
    Strategy:
    1. Pick chunks from first page (title/abstract)
    2. Pick chunks from middle (methodology)
    3. Pick chunks mentioning key entities
    """
    questions = []
    
    # Q1: About the document (from first page)
    first_page_chunks = [c for c in chunks if c['page'] == 1]
    if first_page_chunks:
        questions.append({
            "id": "q1",
            "question": "What is this document about?",
            "expected_page": 1,
            "notes": "Should retrieve title/abstract from page 1"
        })
    
    # Q2: Main contribution/method
    method_keywords = ['method', 'approach', 'algorithm', 'framework', 'technique']
    method_chunks = [c for c in chunks if any(kw in c['text'].lower() for kw in method_keywords)]
    if method_chunks:
        page = method_chunks[0]['page']
        questions.append({
            "id": "q2", 
            "question": "What method or approach is described?",
            "expected_page": page,
            "notes": f"Should retrieve methodology from page {page}"
        })
    
    # Q3: Key findings/results
    result_keywords = ['result', 'finding', 'achieved', 'accuracy', 'performance']
    result_chunks = [c for c in chunks if any(kw in c['text'].lower() for kw in result_keywords)]
    if result_chunks:
        page = result_chunks[0]['page']
        questions.append({
            "id": "q3",
            "question": "What are the main results or findings?",
            "expected_page": page,
            "notes": f"Should retrieve results from page {page}"
        })
    
    # Q4: Problem being solved
    problem_keywords = ['problem', 'challenge', 'issue', 'difficult', 'limitation']
    problem_chunks = [c for c in chunks if any(kw in c['text'].lower() for kw in problem_keywords)]
    if problem_chunks:
        page = problem_chunks[0]['page']
        questions.append({
            "id": "q4",
            "question": "What problem does this document address?",
            "expected_page": page,
            "notes": f"Should retrieve problem statement from page {page}"
        })
    
    # Q5: Specific entity (find most mentioned proper noun)
    # Simple heuristic: words starting with capital letters
    all_text = " ".join([c['text'] for c in chunks[:20]])
    import re
    cap_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', all_text)
    from collections import Counter
    word_counts = Counter(cap_words)
    
    # Filter out common words
    common = ['The', 'This', 'That', 'These', 'Section', 'Figure', 'Table', 'Abstract']
    for c in common:
        word_counts.pop(c, None)
    
    if word_counts:
        top_entity = word_counts.most_common(1)[0][0]
        # Find which page mentions this entity most
        entity_pages = {}
        for c in chunks:
            count = c['text'].count(top_entity)
            if count > 0:
                entity_pages[c['page']] = entity_pages.get(c['page'], 0) + count
        
        if entity_pages:
            best_page = max(entity_pages, key=entity_pages.get)
            questions.append({
                "id": "q5",
                "question": f"What is {top_entity}?",
                "expected_page": best_page,
                "notes": f"Should retrieve information about {top_entity}"
            })
    
    return questions


def save_universal_benchmark(chunks: List[Dict], output_path: str):
    """Generate and save benchmark for any PDF."""
    questions = generate_benchmark_from_chunks(chunks, num_questions=5)
    
    with open(output_path, 'w') as f:
        json.dump(questions, f, indent=2)
    
    print(f"Generated {len(questions)} benchmark questions")
    print(f"Saved to: {output_path}")
    
    for q in questions:
        print(f"  {q['id']}: {q['question']}")
    
    return questions


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.insert(0, '../..')
    from app.ingestion.pdf_loader import load_pdf
    from app.processing.text_splitter import split_pages_into_chunks_semantic
    
    pages = load_pdf("data/raw/sample.pdf")
    chunks = split_pages_into_chunks_semantic(pages)
    
    save_universal_benchmark(chunks, "evaluation/benchmark/auto_generated.json")
