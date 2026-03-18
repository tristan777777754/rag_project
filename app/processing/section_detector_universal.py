"""
Universal Section Detection
Auto-detects document structure without hardcoded academic paper assumptions.
"""
import re
from typing import List, Dict, Any


def detect_sections_universal(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Universal section detection that works for any document type.
    
    Strategy:
    1. Analyze first page for title/intro content
    2. Look for section headers throughout document
    3. Use heuristics based on formatting patterns
    
    Returns pages with detected section metadata.
    """
    # Collect potential section headers from all pages
    all_headers = []
    for page in pages:
        lines = page['text'].split('\n')[:10]  # First 10 lines
        for line in lines:
            line = line.strip()
            # Header patterns: "1. Introduction", "Introduction", "## Methods"
            if _is_likely_header(line):
                all_headers.append((line.lower(), page['page']))
    
    # Detect document type based on header patterns
    doc_type = _detect_document_type(all_headers)
    
    # Apply section detection based on document type
    for page in pages:
        page['section'] = _classify_page_universal(page, doc_type, all_headers)
    
    return pages


def _is_likely_header(line: str) -> bool:
    """Check if a line looks like a section header."""
    if len(line) < 3 or len(line) > 100:
        return False
    
    # Numbered headers: "1. Introduction", "1 Introduction", "1.1 Method"
    if re.match(r'^\d+[\.\s]\s*\w+', line):
        return True
    
    # All caps headers (common in reports)
    if line.isupper() and len(line.split()) <= 5:
        return True
    
    # Markdown headers
    if line.startswith('#'):
        return True
    
    # Common section keywords
    section_keywords = ['introduction', 'background', 'method', 'results', 
                       'discussion', 'conclusion', 'summary', 'abstract',
                       'overview', 'approach', 'implementation']
    if any(kw in line.lower() for kw in section_keywords):
        return True
    
    return False


def _detect_document_type(headers: List[tuple]) -> str:
    """
    Detect document type based on header patterns.
    Returns: 'academic', 'report', 'slides', 'generic'
    """
    header_texts = [h[0] for h in headers]
    
    # Academic paper indicators
    academic_kws = ['abstract', 'introduction', 'methodology', 'references', 
                   'bibliography', 'theorem', 'proof', 'lemma']
    academic_score = sum(1 for h in header_texts for kw in academic_kws if kw in h)
    
    # Report indicators  
    report_kws = ['executive summary', 'background', 'findings', 'recommendations']
    report_score = sum(1 for h in header_texts for kw in report_kws if kw in h)
    
    if academic_score >= 2:
        return 'academic'
    elif report_score >= 2:
        return 'report'
    elif len(headers) < 3:
        return 'slides'  # Few headers = likely slides
    else:
        return 'generic'


def _classify_page_universal(page: Dict, doc_type: str, all_headers: List[tuple]) -> str:
    """Classify a single page based on document type and content."""
    text = page['text'].lower()
    page_num = page['page']
    
    # First page often contains title/abstract/overview
    if page_num == 1:
        if any(kw in text[:1000] for kw in ['abstract', 'summary', 'overview']):
            return 'overview'
        return 'start'
    
    # Check for explicit headers on this page
    lines = page['text'].split('\n')[:5]
    for line in lines:
        line_clean = line.strip().lower()
        if 'conclusion' in line_clean or 'summary' in line_clean:
            return 'conclusion'
        if 'reference' in line_clean or 'bibliography' in line_clean:
            return 'references'
        if 'acknowledgment' in line_clean or 'thank' in line_clean:
            return 'acknowledgments'
        if 'introduction' in line_clean or 'background' in line_clean:
            return 'introduction'
        if any(kw in line_clean for kw in ['method', 'approach', 'implementation']):
            return 'methodology'
        if any(kw in line_clean for kw in ['result', 'finding', 'analysis']):
            return 'results'
    
    # Last few pages often have references
    total_pages = max(h[1] for h in all_headers) if all_headers else page_num
    if page_num > total_pages - 2:
        # Check for reference patterns
        if sum(1 for kw in ['et al', 'pp.', 'vol.', 'doi'] if kw in text) >= 2:
            return 'references'
    
    return 'body'


# Backward compatibility wrapper
def detect_section(text: str, page_num: int) -> str:
    """
    Backward-compatible wrapper that uses universal detection.
    Falls back to academic detection if needed.
    """
    # Try academic detection first (more precise for papers)
    from app.processing.text_splitter import detect_section as academic_detect
    try:
        return academic_detect(text, page_num)
    except:
        pass
    
    # Fallback to universal detection
    page = {'text': text, 'page': page_num}
    result = _classify_page_universal(page, 'generic', [])
    return result
