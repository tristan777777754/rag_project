"""
Retrieval Evaluation Framework
Simple scoring system for retrieval quality assessment.
"""
import json
from typing import List, Dict, Any


class RetrievalEvaluator:
    """Evaluates retrieval quality based on simple practical metrics."""
    
    def __init__(self, benchmark_path: str):
        with open(benchmark_path, 'r') as f:
            self.benchmark = json.load(f)
    
    def score_retrieval(self, question_id: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Score retrieval for a single question.
        
        Returns:
            dict with scores for: relevance, important_section_hit, noise_level, coverage
        """
        question_data = next(q for q in self.benchmark if q['id'] == question_id)
        
        # 1. Important Section Hit Rate (critical metric)
        expected_page = question_data.get('expected_page', 1)
        pages_retrieved = [chunk.get('page', 0) for chunk in retrieved_chunks]
        hit_important = expected_page in pages_retrieved
        
        # 2. Noise Detection
        noise_indicators = ['references', 'bibliography', 'acknowledgments', 'appendix']
        noisy_chunks = 0
        for chunk in retrieved_chunks:
            text_lower = chunk.get('text', '').lower()
            if any(ind in text_lower[:200] for ind in noise_indicators):
                noisy_chunks += 1
            # Tiny fragments check
            if len(chunk.get('text', '')) < 50:
                noisy_chunks += 1
        
        noise_ratio = noisy_chunks / len(retrieved_chunks) if retrieved_chunks else 1.0
        noise_score = max(0, 1 - noise_ratio)  # 1 = no noise, 0 = all noise
        
        # 3. Abstract/Intro Coverage (for page 1 questions)
        abstract_keywords = ['abstract', 'this paper introduces', 'we propose', 'in this paper']
        intro_keywords = ['introduction', 'this section', 'background']
        
        has_abstract_content = any(
            any(kw in chunk.get('text', '').lower()[:500] for kw in abstract_keywords)
            for chunk in retrieved_chunks
        )
        
        # 4. Relevance heuristic (check for expected topic keywords)
        expected_topic = question_data.get('expected_topic', '').lower()
        topic_keywords = [kw for kw in expected_topic.split() if len(kw) > 4][:5]  # Top 5 longer keywords
        
        relevance_hits = 0
        for chunk in retrieved_chunks:
            chunk_text = chunk.get('text', '').lower()
            if any(kw in chunk_text for kw in topic_keywords):
                relevance_hits += 1
        
        relevance_score = relevance_hits / len(retrieved_chunks) if retrieved_chunks else 0
        
        # Overall score (weighted)
        # Important section hit is critical (0.4), relevance (0.3), noise (0.2), abstract content (0.1)
        overall = (
            (0.4 if hit_important else 0) +
            (relevance_score * 0.3) +
            (noise_score * 0.2) +
            (0.1 if has_abstract_content else 0)
        )
        
        return {
            "question_id": question_id,
            "question": question_data['question'],
            "important_section_hit": hit_important,
            "expected_page": expected_page,
            "pages_retrieved": pages_retrieved,
            "noise_score": round(noise_score, 2),
            "relevance_score": round(relevance_score, 2),
            "has_abstract_content": has_abstract_content,
            "overall_score": round(overall, 2)
        }
    
    def evaluate_all(self, retrieval_fn) -> Dict[str, Any]:
        """
        Run evaluation on all benchmark questions.
        retrieval_fn: function that takes a query string and returns list of chunks
        """
        results = []
        total_score = 0
        important_hits = 0
        
        for question_data in self.benchmark:
            qid = question_data['id']
            query = question_data['question']
            
            # Get retrieved chunks
            chunks = retrieval_fn(query)
            
            # Score
            score = self.score_retrieval(qid, chunks)
            results.append(score)
            
            total_score += score['overall_score']
            if score['important_section_hit']:
                important_hits += 1
        
        avg_score = total_score / len(self.benchmark) if self.benchmark else 0
        important_hit_rate = important_hits / len(self.benchmark) if self.benchmark else 0
        
        return {
            "results": results,
            "summary": {
                "total_questions": len(self.benchmark),
                "avg_score": round(avg_score, 2),
                "important_section_hit_rate": round(important_hit_rate, 2),
                "important_hits": important_hits,
                "misses": len(self.benchmark) - important_hits
            }
        }


def format_report(evaluation_result: Dict[str, Any]) -> str:
    """Format evaluation results as a readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append("RETRIEVAL EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    summary = evaluation_result['summary']
    lines.append(f"Total Questions: {summary['total_questions']}")
    lines.append(f"Average Score: {summary['avg_score']:.2f} / 1.0")
    lines.append(f"Important Section Hit Rate: {summary['important_section_hit_rate']:.0%}")
    lines.append(f"Hits: {summary['important_hits']} | Misses: {summary['misses']}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("DETAILED RESULTS")
    lines.append("-" * 60)
    lines.append("")
    
    for r in evaluation_result['results']:
        status = "✓ HIT" if r['important_section_hit'] else "✗ MISS"
        lines.append(f"[{status}] Q{r['question_id']}: {r['question'][:50]}...")
        lines.append(f"         Expected page: {r['expected_page']} | Retrieved: {r['pages_retrieved']}")
        lines.append(f"         Score: {r['overall_score']:.2f} | Relevance: {r['relevance_score']:.2f} | Noise: {r['noise_score']:.2f}")
        lines.append("")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Simple test
    evaluator = RetrievalEvaluator("evaluation/benchmark/questions.json")
    print(f"Loaded {len(evaluator.benchmark)} benchmark questions")
    print("\nSample question:")
    print(json.dumps(evaluator.benchmark[0], indent=2))
