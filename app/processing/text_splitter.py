"""
Improved Semantic Chunking (Finance-aware)
Now uses page-level section detection instead of chunk-only detection.
"""

import re
from typing import List, Dict, Any

# 🔥 NEW: import finance section detector
from app.processing.finance_section_detector import build_page_section_map


def detect_section(text: str, page_num: int) -> str:
    """
    Fallback section detection (kept for safety).
    """
    text_lower = text.lower().strip()
    first_500 = text_lower[:500]

    if "abstract" in first_500:
        return "abstract"
    if "introduction" in first_500:
        return "introduction"
    if "method" in first_500 or "model" in first_500:
        return "methodology"
    if "result" in first_500 or "performance" in first_500:
        return "results"
    if "conclusion" in first_500:
        return "conclusion"
    if "reference" in first_500:
        return "references"

    return "body"


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex.
    """
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def split_pages_into_chunks_semantic(
    pages: List[Dict[str, Any]],
    max_chunk_size: int = 800,
    min_chunk_size: int = 200,
    overlap_sentences: int = 1
) -> List[Dict[str, Any]]:
    """
    Finance-aware semantic chunking.
    Uses page-level section detection + fallback.
    """

    chunks = []

    # 🔥 NEW: build page-level section map
    page_section_map = build_page_section_map(pages)

    for page in pages:
        text = page["text"]
        page_num = page["page"]

        # Skip tiny pages (except page 1)
        if len(text.strip()) < min_chunk_size:
            if page_num == 1 and len(text.strip()) > 50:
                section = page_section_map.get(page_num, "body")

                if section == "body":
                    section = detect_section(text, page_num)

                chunks.append({
                    "text": text.strip(),
                    "page": page_num,
                    "section": section,
                    "position": "start"
                })
            continue

        sentences = split_into_sentences(text)
        if not sentences:
            continue

        current_chunk_sentences = []
        current_length = 0

        for i, sentence in enumerate(sentences):
            sentence_len = len(sentence)

            if current_length + sentence_len > max_chunk_size and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)

                if len(chunk_text) >= min_chunk_size:
                    # ✅ FIXED: use page-level section first
                    section = page_section_map.get(page_num, "body")

                    # fallback if still body
                    if section == "body":
                        section = detect_section(chunk_text, page_num)

                    position = "start" if page_num == 1 and not chunks else "body"

                    chunks.append({
                        "text": chunk_text,
                        "page": page_num,
                        "section": section,
                        "position": position
                    })

                # overlap
                if overlap_sentences > 0:
                    current_chunk_sentences = current_chunk_sentences[-overlap_sentences:]
                    current_length = sum(len(s) for s in current_chunk_sentences)
                else:
                    current_chunk_sentences = []
                    current_length = 0

            current_chunk_sentences.append(sentence)
            current_length += sentence_len + 1

        # last chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)

            if len(chunk_text) >= min_chunk_size:
                section = page_section_map.get(page_num, "body")

                if section == "body":
                    section = detect_section(chunk_text, page_num)

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
    Filter out noisy chunks.
    """

    filtered = []

    for chunk in chunks:
        text = chunk["text"]
        section = chunk.get("section", "body")

        # Always keep abstract
        if section == "abstract":
            filtered.append(chunk)
            continue

        # Remove useless sections
        if section in ("references", "acknowledgments"):
            continue

        if len(text.strip()) < 100:
            continue

        # table-like detection
        lines = text.split('\n')
        if len(lines) > 5:
            short_lines = sum(1 for l in lines if len(l.strip()) < 30)
            if short_lines / len(lines) > 0.7:
                continue

        filtered.append(chunk)

    return filtered


# backward compatible
def split_pages_into_chunks(pages, chunk_size=800, overlap=100):
    return split_pages_into_chunks_semantic(
        pages,
        max_chunk_size=chunk_size,
        overlap_sentences=1 if overlap > 0 else 0
    )