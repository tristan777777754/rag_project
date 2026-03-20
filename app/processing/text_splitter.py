"""
Improved Semantic Chunking (Finance-aware)
STRICT page-level section detection (no noisy fallback override)
"""

import re
from typing import List, Dict, Any
from app.processing.finance_section_detector import build_page_section_map


def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def split_pages_into_chunks_semantic(
    pages: List[Dict[str, Any]],
    max_chunk_size: int = 800,
    min_chunk_size: int = 200,
    overlap_sentences: int = 1
) -> List[Dict[str, Any]]:

    chunks = []

  
    page_section_map = build_page_section_map(pages)

    print("\n=== DEBUG: PAGE SECTION MAP ===")
    for p in sorted(page_section_map)[:15]:
        print(p, page_section_map[p])

    for page in pages:
        text = page["text"]
        page_num = page["page"]

        page_section = page_section_map.get(page_num, "body")

        # skip very small pages
        if len(text.strip()) < min_chunk_size:
            if page_num == 1 and len(text.strip()) > 50:
                chunks.append({
                    "text": text.strip(),
                    "page": page_num,
                    "section": page_section,
                    "position": "start"
                })
            continue

        sentences = split_into_sentences(text)
        if not sentences:
            continue

        current_chunk_sentences = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_length + sentence_len > max_chunk_size and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)

                if len(chunk_text) >= min_chunk_size:
                    chunks.append({
                        "text": chunk_text,
                        "page": page_num,
                        "section": page_section,  # ✅ ONLY use page-level
                        "position": "start" if page_num == 1 and not chunks else "body"
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
                chunks.append({
                    "text": chunk_text,
                    "page": page_num,
                    "section": page_section,
                    "position": "start" if page_num == 1 and not chunks else "body"
                })

    return chunks


def filter_noisy_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

    filtered = []

    for chunk in chunks:
        text = chunk["text"]
        section = chunk.get("section", "body")

        # keep abstract always
        if section == "abstract":
            filtered.append(chunk)
            continue

        # remove references
        if section in ("references", "acknowledgments"):
            continue

        if len(text.strip()) < 100:
            continue

        # remove table-like chunks
        lines = text.split('\n')
        if len(lines) > 5:
            short_lines = sum(1 for l in lines if len(l.strip()) < 30)
            if short_lines / len(lines) > 0.7:
                continue

        filtered.append(chunk)

    return filtered


def split_pages_into_chunks(pages, chunk_size=800, overlap=100):
    return split_pages_into_chunks_semantic(
        pages,
        max_chunk_size=chunk_size,
        overlap_sentences=1 if overlap > 0 else 0
    )