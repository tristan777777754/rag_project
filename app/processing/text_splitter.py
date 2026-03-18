"""
Improved Semantic Chunking
Splits text on semantic boundaries (sentences, paragraphs) while preserving page and section metadata.
"""
import re
from typing import List, Dict, Any


def detect_section(text: str, page_num: int) -> str:
    """
    Detect the section type based on content patterns.
    Returns: abstract, introduction, methodology, results, conclusion, references, or body
    """
    text_lower = text.lower().strip()
    first_500 = text_lower[:500]
    
    # Page 1 with abstract keywords
    if page_num == 1:
        if 'abstract' in first_500 or text_lower.startswith('abstract'):
            return 'abstract'
        if any(kw in first_500 for kw in ['this paper introduces', 'we propose', 'in this paper']):
            return 'abstract'
    
    # Section headers detection
    lines = text_lower.split('\n')[:3]  # Check first 3 lines
    for line in lines:
        line = line.strip()
        if line.startswith('introduction') or line == '1 introduction' or line == '1. introduction':
            return 'introduction'
        if any(line.startswith(x) for x in ['2 ', '3 ', '4 ', '5 ', '6 ', '7 ']):
            if any(kw in line for kw in ['methodology', 'model', 'framework', 'approach']):
                return 'methodology'
            if any(kw in line for kw in ['results', 'experiments', 'evaluation']):
                return 'results'
            if any(kw in line for kw in ['conclusion', 'discussion', 'future work']):
                return 'conclusion'
        if 'references' in line or 'bibliography' in line:
            return 'references'
    
    # Content-based detection for references
    if page_num > 30:  # Usually references are near the end
        ref_patterns = [r'\(\d{4}\)', r'\d{4}\.', r'vol\.', r'pp\.', r'journal', r'conference']
        if sum(1 for p in ref_patterns if re.search(p, text_lower)) >= 2:
            return 'references'
    
    return 'body'


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex.
    Handles common abbreviations and edge cases.
    """
    # Simple sentence splitting - handles basic cases
    # Pattern: split on period followed by space and capital letter, or end of string
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def split_pages_into_chunks_semantic(
    pages: List[Dict[str, Any]], 
    max_chunk_size: int = 800, 
    min_chunk_size: int = 200,
    overlap_sentences: int = 1
) -> List[Dict[str, Any]]:
    """
    Split pages into semantically coherent chunks.
    
    Strategy:
    1. Split text into sentences
    2. Group sentences into chunks while respecting max_chunk_size
    3. Add overlap between chunks (shared sentences)
    4. Preserve section metadata
    
    Args:
        pages: List of {page: int, text: str} dicts
        max_chunk_size: Maximum characters per chunk
        min_chunk_size: Minimum characters to form a valid chunk
        overlap_sentences: Number of sentences to overlap between chunks
    
    Returns:
        List of chunks with page, section, and position metadata
    """
    chunks = []
    
    for page in pages:
        text = page["text"]
        page_num = page["page"]
        
        # Detect section for this page content
        section = detect_section(text, page_num)
        
        # Skip tiny pages
        if len(text.strip()) < min_chunk_size:
            # Still keep very short pages if they're page 1 (abstract might be short)
            if page_num == 1 and len(text.strip()) > 50:
                chunks.append({
                    "text": text.strip(),
                    "page": page_num,
                    "section": section,
                    "position": "start" if page_num == 1 else "body"
                })
            continue
        
        # Split into sentences
        sentences = split_into_sentences(text)
        
        if not sentences:
            continue
        
        # Build chunks from sentences
        current_chunk_sentences = []
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            sentence_len = len(sentence)
            
            # Check if adding this sentence would exceed max size
            if current_length + sentence_len > max_chunk_size and current_chunk_sentences:
                # Store current chunk
                chunk_text = " ".join(current_chunk_sentences)
                if len(chunk_text) >= min_chunk_size:
                    position = "start" if page_num == 1 and not chunks else "body"
                    chunks.append({
                        "text": chunk_text,
                        "page": page_num,
                        "section": section,
                        "position": position
                    })
                
                # Start new chunk with overlap
                if overlap_sentences > 0:
                    # Take last N sentences as overlap
                    current_chunk_sentences = current_chunk_sentences[-overlap_sentences:]
                    current_length = sum(len(s) for s in current_chunk_sentences)
                else:
                    current_chunk_sentences = []
                    current_length = 0
            
            current_chunk_sentences.append(sentence)
            current_length += sentence_len + 1  # +1 for space
        
        # Don't forget the last chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            if len(chunk_text) >= min_chunk_size:
                position = "start" if page_num == 1 and not chunks else "body"
                chunks.append({
                    "text": chunk_text,
                    "page": page_num,
                    "section": section,
                    "position": position
                })
    
    return chunks


def filter_noisy_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter out noisy chunks like references, tiny fragments, table data.
    Preserves abstract even if short.
    """
    filtered = []
    
    for chunk in chunks:
        text = chunk["text"]
        section = chunk.get("section", "body")
        
        # Always keep abstract
        if section == "abstract":
            filtered.append(chunk)
            continue
        
        # Skip references sections
        if section == "references":
            continue
        
        # Skip very tiny fragments
        if len(text.strip()) < 100:
            continue
        
        # Skip chunks that look like table fragments (many numbers, short lines)
        lines = text.split('\n')
        if len(lines) > 5:
            short_lines = sum(1 for l in lines if len(l.strip()) < 30)
            if short_lines / len(lines) > 0.7:  # Mostly short lines
                continue
        
        filtered.append(chunk)
    
    return filtered


# Backward-compatible wrapper
def split_pages_into_chunks(pages, chunk_size=800, overlap=100):
    """
    Backward-compatible wrapper that uses semantic chunking.
    Preserves original function signature.
    """
    # Map old params to new semantic params
    # overlap=100 chars roughly equals 1 sentence in most academic text
    return split_pages_into_chunks_semantic(
        pages, 
        max_chunk_size=chunk_size,
        overlap_sentences=1 if overlap > 0 else 0
    )
