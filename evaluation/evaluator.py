"""
Retrieval Evaluation Framework

Supports both the legacy benchmark schema and the newer domain-specific schema
for factor investing papers.
"""
import json
from typing import List, Dict, Any


class RetrievalEvaluator:
    """Evaluates retrieval quality using section hits, relevance, and noise."""

    def __init__(self, benchmark_path: str):
        with open(benchmark_path, 'r') as f:
            self.benchmark = json.load(f)

    def _get_query_text(self, question_data: Dict[str, Any]) -> str:
        return question_data.get('query') or question_data.get('question', '')

    def _get_expected_sections(self, question_data: Dict[str, Any]) -> List[str]:
        gold_section = question_data.get('gold_section')
        expected_section = question_data.get('expected_section')

        if isinstance(gold_section, list):
            return [s.lower() for s in gold_section]
        if isinstance(gold_section, str):
            return [s.strip().lower() for s in gold_section.split(',')]
        if isinstance(expected_section, list):
            return [s.lower() for s in expected_section]
        if isinstance(expected_section, str):
            return [s.strip().lower() for s in expected_section.split(',')]

        return []

    def _get_expected_answers(self, question_data: Dict[str, Any]) -> List[str]:
        answers = []
        if question_data.get('gold_answer'):
            answers.append(str(question_data['gold_answer']).lower())
        if question_data.get('expected_topic'):
            answers.append(str(question_data['expected_topic']).lower())
        return answers

    def score_retrieval(self, question_id: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Score retrieval for a single question."""
        question_data = next(q for q in self.benchmark if q['id'] == question_id)

        expected_page = question_data.get('expected_page')
        expected_sections = self._get_expected_sections(question_data)
        expected_answers = self._get_expected_answers(question_data)
        query_type = question_data.get('query_type', 'unknown')

        pages_retrieved = [chunk.get('page', 0) for chunk in retrieved_chunks]
        sections_retrieved = [str(chunk.get('section', 'body')).lower() for chunk in retrieved_chunks]

        # Section-aware hit metric for the new benchmark, with page fallback for
        # the legacy benchmark.
        section_hit = any(sec in expected_sections for sec in sections_retrieved) if expected_sections else False
        page_hit = expected_page in pages_retrieved if expected_page is not None else False
        hit_important = section_hit or page_hit

        # Noise detection.
        noise_indicators = ['references', 'bibliography', 'acknowledgments']
        noisy_chunks = 0
        for chunk in retrieved_chunks:
            text_lower = chunk.get('text', '').lower()
            section = str(chunk.get('section', '')).lower()
            if section in {'references', 'acknowledgments'}:
                noisy_chunks += 1
            elif any(ind in text_lower[:200] for ind in noise_indicators):
                noisy_chunks += 1
            elif len(chunk.get('text', '')) < 50:
                noisy_chunks += 1

        noise_ratio = noisy_chunks / len(retrieved_chunks) if retrieved_chunks else 1.0
        noise_score = max(0, 1 - noise_ratio)

        # Relevance heuristic: look for expected answer keywords.
        relevance_hits = 0
        keyword_bank = []
        for answer in expected_answers:
            keyword_bank.extend([kw for kw in answer.split() if len(kw) > 4][:8])
        keyword_bank = list(dict.fromkeys(keyword_bank))[:10]

        for chunk in retrieved_chunks:
            chunk_text = chunk.get('text', '').lower()
            if keyword_bank and any(kw in chunk_text for kw in keyword_bank):
                relevance_hits += 1

        relevance_score = relevance_hits / len(retrieved_chunks) if retrieved_chunks else 0

        # Structure alignment bonus: reward retrievals whose chunk sections match
        # the benchmark's expected section labels.
        structure_score = 1.0 if section_hit else 0.0

        overall = (
            (0.4 if hit_important else 0.0) +
            (relevance_score * 0.3) +
            (noise_score * 0.2) +
            (structure_score * 0.1)
        )

        return {
            'question_id': question_id,
            'query': self._get_query_text(question_data),
            'query_type': query_type,
            'important_section_hit': hit_important,
            'section_hit': section_hit,
            'page_hit': page_hit,
            'expected_page': expected_page,
            'expected_sections': expected_sections,
            'pages_retrieved': pages_retrieved,
            'sections_retrieved': sections_retrieved,
            'noise_score': round(noise_score, 2),
            'relevance_score': round(relevance_score, 2),
            'structure_score': round(structure_score, 2),
            'overall_score': round(overall, 2),
        }

    def evaluate_all(self, retrieval_fn) -> Dict[str, Any]:
        """Run evaluation on all benchmark queries."""
        results = []
        total_score = 0
        important_hits = 0
        query_type_breakdown: Dict[str, Dict[str, float]] = {}

        for question_data in self.benchmark:
            qid = question_data['id']
            query = self._get_query_text(question_data)

            chunks = retrieval_fn(query)
            score = self.score_retrieval(qid, chunks)
            results.append(score)

            total_score += score['overall_score']
            if score['important_section_hit']:
                important_hits += 1

            qtype = score['query_type']
            query_type_breakdown.setdefault(qtype, {'count': 0, 'hits': 0, 'score_sum': 0.0})
            query_type_breakdown[qtype]['count'] += 1
            query_type_breakdown[qtype]['score_sum'] += score['overall_score']
            if score['important_section_hit']:
                query_type_breakdown[qtype]['hits'] += 1

        avg_score = total_score / len(self.benchmark) if self.benchmark else 0
        important_hit_rate = important_hits / len(self.benchmark) if self.benchmark else 0

        formatted_breakdown = {}
        for qtype, stats in query_type_breakdown.items():
            count = stats['count']
            formatted_breakdown[qtype] = {
                'count': count,
                'hit_rate': round(stats['hits'] / count, 2) if count else 0,
                'avg_score': round(stats['score_sum'] / count, 2) if count else 0,
            }

        return {
            'results': results,
            'summary': {
                'total_questions': len(self.benchmark),
                'avg_score': round(avg_score, 2),
                'important_section_hit_rate': round(important_hit_rate, 2),
                'important_hits': important_hits,
                'misses': len(self.benchmark) - important_hits,
                'query_type_breakdown': formatted_breakdown,
            }
        }


def format_report(evaluation_result: Dict[str, Any]) -> str:
    """Format evaluation results as a readable report."""
    lines = []
    lines.append('=' * 60)
    lines.append('RETRIEVAL EVALUATION REPORT')
    lines.append('=' * 60)
    lines.append('')

    summary = evaluation_result['summary']
    lines.append(f"Total Questions: {summary['total_questions']}")
    lines.append(f"Average Score: {summary['avg_score']:.2f} / 1.0")
    lines.append(f"Important Section Hit Rate: {summary['important_section_hit_rate']:.0%}")
    lines.append(f"Hits: {summary['important_hits']} | Misses: {summary['misses']}")
    lines.append('')

    if summary.get('query_type_breakdown'):
        lines.append('Query Type Breakdown:')
        for qtype, stats in summary['query_type_breakdown'].items():
            lines.append(
                f"  - {qtype}: count={stats['count']}, hit_rate={stats['hit_rate']:.0%}, avg_score={stats['avg_score']:.2f}"
            )
        lines.append('')

    lines.append('-' * 60)
    lines.append('DETAILED RESULTS')
    lines.append('-' * 60)
    lines.append('')

    for r in evaluation_result['results']:
        status = '✓ HIT' if r['important_section_hit'] else '✗ MISS'
        lines.append(f"[{status}] {r['question_id']} ({r['query_type']}): {r['query'][:60]}...")
        lines.append(f"         Expected sections: {r['expected_sections']} | Retrieved: {r['sections_retrieved'][:5]}")
        if r['expected_page'] is not None:
            lines.append(f"         Expected page: {r['expected_page']} | Retrieved pages: {r['pages_retrieved']}")
        lines.append(
            f"         Score: {r['overall_score']:.2f} | Relevance: {r['relevance_score']:.2f} | Noise: {r['noise_score']:.2f} | Structure: {r['structure_score']:.2f}"
        )
        lines.append('')

    return '\n'.join(lines)


if __name__ == '__main__':
    evaluator = RetrievalEvaluator('evaluation/benchmark/questions.json')
    print(f"Loaded {len(evaluator.benchmark)} benchmark questions")
    print('\nSample question:')
    print(json.dumps(evaluator.benchmark[0], indent=2))
