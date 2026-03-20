"""
Header-driven semantic chunking for finance papers.

Design:
- Detect page sections using a document state machine.
- Separate top-level section from subsection topic.
- Chunks inherit locked page metadata; they do not re-guess section from local text.
"""
import re
from typing import List, Dict, Any, Optional, Tuple


CANONICAL_SECTION_MAP = {
    "front_matter": "front_matter",
    "abstract": "abstract",
    "introduction": "introduction",
    "methodology_portfolio": "methodology",
    "data_sample": "methodology",
    "performance": "results",
    "robustness": "results",
    "discussion": "body",
    "conclusion": "conclusion",
    "references": "references",
    "acknowledgments": "acknowledgments",
    "body": "body",
}

# Only these are allowed to change the top-level section state.
MAJOR_HEADER_TO_TOPIC = {
    "abstract": "abstract",
    "summary": "introduction",
    "questions": "introduction",
    "introduction": "introduction",
    "structural estimation": "methodology_portfolio",
    "quantitative theories": "methodology_portfolio",
    "conclusion": "conclusion",
    "references": "references",
    "bibliography": "references",
    "acknowledgments": "acknowledgments",
    "acknowledgements": "acknowledgments",
    "implications": "discussion",
}

# These are subsection topics. They should not necessarily change the top-level section.
SUBSECTION_TOPIC_MAP = {
    "summary": "introduction",
    "questions": "introduction",
    "mechanisms": "introduction",
    "methods": "introduction",
    "factor models": "introduction",
    "intuition": "introduction",
    "subsequent work": "introduction",
    "structural estimation": "methodology_portfolio",
    "quantitative theories": "methodology_portfolio",
    "data": "data_sample",
    "sample": "data_sample",
    "dataset": "data_sample",
    "empirical setting": "data_sample",
    "performance": "performance",
    "results": "performance",
    "robustness": "robustness",
    "additional tests": "robustness",
    "sensitivity analysis": "robustness",
    "implications": "discussion",
    "complementarity with the consumption capm": "discussion",
    "an emh counterrevolution to behavioral finance": "discussion",
    "how i defend fama": "discussion",
    "security analysis within efficient markets": "discussion",
    "rational expectations economics": "discussion",
    "challenges": "discussion",
    "a risky mechanism of momentum": "discussion",
    "other asset classes": "discussion",
    "conclusion": "conclusion",
    "references": "references",
}


def _normalize_line(line: str) -> str:
    line = line.strip().lower()
    line = line.replace("ﬁ", "fi").replace("ﬂ", "fl")
    line = re.sub(r"\s+", " ", line)
    return line.strip(" .:-")


def _looks_like_header(line: str) -> bool:
    normalized = _normalize_line(line)
    if not normalized or len(normalized) > 90:
        return False
    if re.fullmatch(r"\d+", normalized):
        return False
    if re.search(r"[.!?;:]", normalized):
        return False
    words = normalized.split()
    if len(words) > 10:
        return False
    boilerplate = {
        "nber working paper series",
        "q-factors and investment capm",
        "lu zhang",
        "working paper 26538",
        "national bureau of economic research",
        "december 2019",
        "jel no e13,e22,e32,e44,g12,g14,g31,m41",
    }
    if normalized in boilerplate:
        return False
    return True


def _match_header_topic(line: str) -> Optional[str]:
    normalized = _normalize_line(line)
    if not _looks_like_header(normalized):
        return None
    return SUBSECTION_TOPIC_MAP.get(normalized)


def _match_major_topic(line: str) -> Optional[str]:
    normalized = _normalize_line(line)
    if not _looks_like_header(normalized):
        return None
    return MAJOR_HEADER_TO_TOPIC.get(normalized)


def _page_preview_lines(text: str, limit: int = 40) -> List[str]:
    return [line.strip() for line in text.split("\n")[:limit] if line.strip()]


def detect_page_sections(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    detected_pages: List[Dict[str, Any]] = []
    current_topic = "front_matter"
    current_subtopic = "front_matter"
    abstract_seen = False
    total_pages = len(pages)

    for page in pages:
        page_num = page["page"]
        text = page["text"]
        early_lines = _page_preview_lines(text)

        matched_major = None
        matched_sub = None

        for line in early_lines:
            if matched_major is None:
                matched_major = _match_major_topic(line)
            if matched_sub is None:
                matched_sub = _match_header_topic(line)
            if matched_major and matched_sub:
                break

        page_topic = current_topic
        page_subtopic = current_subtopic

        if matched_major:
            page_topic = matched_major
            current_topic = matched_major
            page_subtopic = matched_sub or matched_major
            current_subtopic = page_subtopic
            if matched_major == "abstract":
                abstract_seen = True
        else:
            # Limited fallback only for front matter / abstract / references.
            if page_num == 1:
                page_topic = "front_matter"
                page_subtopic = "front_matter"
                current_topic = page_topic
                current_subtopic = page_subtopic
            elif not abstract_seen and page_num <= 3:
                joined = " ".join(early_lines[:10]).lower()
                if "abstract" in joined:
                    page_topic = "abstract"
                    page_subtopic = "abstract"
                    current_topic = page_topic
                    current_subtopic = page_subtopic
                    abstract_seen = True
                elif current_topic == "front_matter" and len(text.strip()) > 400:
                    page_topic = "abstract"
                    page_subtopic = "abstract"
                    current_topic = page_topic
                    current_subtopic = page_subtopic
                    abstract_seen = True
                elif matched_sub:
                    page_subtopic = matched_sub
                    current_subtopic = matched_sub
            elif page_num >= total_pages - 5:
                joined = " ".join(early_lines[:12]).lower()
                reference_markers = ["references", "bibliography", "journal of", "review of", "econometrica"]
                if current_topic == "references" or sum(marker in joined for marker in reference_markers) >= 2:
                    page_topic = "references"
                    page_subtopic = "references"
                    current_topic = page_topic
                    current_subtopic = page_subtopic
                elif matched_sub:
                    page_subtopic = matched_sub
                    current_subtopic = matched_sub
            else:
                if matched_sub:
                    page_subtopic = matched_sub
                    current_subtopic = matched_sub

        enriched = dict(page)
        enriched["section_topic"] = page_topic
        enriched["subsection_topic"] = page_subtopic
        enriched["section"] = CANONICAL_SECTION_MAP.get(page_topic, "body")
        detected_pages.append(enriched)

    return detected_pages


def detect_section(text: str, page_num: int) -> str:
    lines = _page_preview_lines(text)
    if page_num == 1:
        return "front_matter"

    for line in lines:
        topic = _match_major_topic(line)
        if topic:
            return CANONICAL_SECTION_MAP.get(topic, "body")

    joined = " ".join(lines[:8]).lower()
    if page_num <= 3 and "abstract" in joined:
        return "abstract"
    if "references" in joined or "bibliography" in joined:
        return "references"
    return "body"


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
    pages_with_sections = detect_page_sections(pages)

    for page in pages_with_sections:
        text = page["text"]
        page_num = page["page"]
        section = page.get("section", "body")
        section_topic = page.get("section_topic", section)
        subsection_topic = page.get("subsection_topic", section_topic)

        if len(text.strip()) < min_chunk_size:
            if page_num <= 2 and len(text.strip()) > 50:
                chunks.append({
                    "text": text.strip(),
                    "page": page_num,
                    "section": section,
                    "section_topic": section_topic,
                    "subsection_topic": subsection_topic,
                    "position": "start" if page_num <= 2 else "body"
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
                        "section": section,
                        "section_topic": section_topic,
                        "subsection_topic": subsection_topic,
                        "position": "start" if page_num <= 2 and not chunks else "body"
                    })
                if overlap_sentences > 0:
                    current_chunk_sentences = current_chunk_sentences[-overlap_sentences:]
                    current_length = sum(len(s) for s in current_chunk_sentences)
                else:
                    current_chunk_sentences = []
                    current_length = 0

            current_chunk_sentences.append(sentence)
            current_length += sentence_len + 1

        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            if len(chunk_text) >= min_chunk_size:
                chunks.append({
                    "text": chunk_text,
                    "page": page_num,
                    "section": section,
                    "section_topic": section_topic,
                    "subsection_topic": subsection_topic,
                    "position": "start" if page_num <= 2 and not chunks else "body"
                })

    return chunks


def filter_noisy_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered = []
    for chunk in chunks:
        text = chunk["text"]
        section = chunk.get("section", "body")
        if section == "abstract":
            filtered.append(chunk)
            continue
        if section in ("references", "acknowledgments"):
            continue
        if len(text.strip()) < 100:
            continue
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
