"""
Header-driven semantic chunking for finance papers.

Design:
- Detect page sections using a document state machine.
- Preserve top-level section and subsection topic separately.
- Chunks inherit locked page metadata; they do not re-guess section locally.
"""
import re
from typing import List, Dict, Any, Optional, Tuple


CANONICAL_SECTION_MAP = {
    "front_matter": "front_matter",
    "abstract": "abstract",
    "introduction": "introduction",
    "related_literature": "introduction",
    "methodology_portfolio": "methodology",
    "data_sample": "methodology",
    "performance": "results",
    "robustness": "results",
    "conclusion": "conclusion",
    "references": "references",
    "appendix": "body",
    "figures_tables": "body",
    "acknowledgments": "acknowledgments",
    "body": "body",
}


def _normalize_line(line: str) -> str:
    line = line.strip().lower()
    line = line.replace("ﬁ", "fi").replace("ﬂ", "fl")
    line = re.sub(r"\s+", " ", line)
    return line.strip(" .:-")


def _page_preview_lines(text: str, limit: int = 60) -> List[str]:
    return [line.strip() for line in text.split("\n")[:limit] if line.strip()]


def _clean_header_text(line: str) -> str:
    normalized = _normalize_line(line)
    normalized = re.sub(r"^(section\s+)?(\d+(\.\d+)*|[ivxlcdm]+|[a-z])\s*[\.)-]?\s+", "", normalized)
    return normalized.strip()


def _header_level(line: str) -> Optional[int]:
    normalized = _normalize_line(line)
    if re.match(r"^\d+\s*[\.)-]?\s+", normalized):
        return 1
    if re.match(r"^\d+\.\d+\s*[\.)-]?\s+", normalized):
        return 2
    if re.match(r"^\d+\.\d+\.\d+\s*[\.)-]?\s+", normalized):
        return 3
    if re.match(r"^[a-z]\s*[\.)-]?\s+", normalized):
        return 1
    return None


def _looks_like_plain_header(line: str) -> bool:
    normalized = _normalize_line(line)
    if not normalized or len(normalized) > 90:
        return False
    if re.search(r"[!?]", normalized):
        return False
    words = normalized.split()
    if len(words) > 10:
        return False
    common = {
        "abstract", "introduction", "related literature", "methodology", "data",
        "conclusion", "references", "bibliography", "appendix", "simulation procedure",
        "tables", "figures", "robustness", "methods", "results"
    }
    return normalized in common


def _extract_candidate_headers(lines: List[str]) -> List[Tuple[int, str]]:
    candidates = []
    for idx, line in enumerate(lines):
        level = _header_level(line)
        if level is not None or _looks_like_plain_header(line):
            candidates.append((idx, line))

        # Handle split headers such as:
        # "4" + "Transaction costs with funding liquidity"
        # "2.1" + "Measuring the effective bid-ask spread from daily prices"
        if re.fullmatch(r"(\d+(\.\d+)*|[ivxlcdm]+|[a-z])", _normalize_line(line)) and idx + 1 < len(lines):
            combined = f"{line} {lines[idx + 1]}"
            candidates.append((idx, combined))
    return candidates


def _topic_from_header_text(header_text: str) -> Optional[str]:
    h = _clean_header_text(header_text)

    exact = {
        "abstract": "abstract",
        "introduction": "introduction",
        "related literature": "related_literature",
        "methodology": "methodology_portfolio",
        "data": "data_sample",
        "conclusion": "conclusion",
        "references": "references",
        "bibliography": "references",
        "simulation procedure": "appendix",
        "tables": "figures_tables",
        "figures": "figures_tables",
    }
    if h in exact:
        return exact[h]

    if any(x in h for x in [
        "measuring the effective bid-ask spread",
        "the model of roll",
        "the bayesian procedure",
        "hasbouck model with funding liquidity",
        "hasbrouck model with funding liquidity",
        "estimation by gibbs sampling",
        "bayes factor",
    ]):
        return "methodology_portfolio"

    if any(x in h for x in [
        "transaction costs with funding liquidity",
        "average transaction costs",
        "the dynamics of transaction costs",
        "transaction costs and firm size",
        "transaction costs and volatility",
        "transaction costs and momentum",
        "transaction costs and flight to quality",
        "gross returns and net returns of anomalies",
        "performance of long-short strategies",
    ]):
        return "performance"

    if any(x in h for x in [
        "robustness",
        "transaction costs with other conditioning financial variables",
        "transaction costs and the vix",
        "transaction costs and tail risk",
        "the lesmond et al",
        "lot model",
        "cost-mitigating trading strategies",
    ]):
        return "robustness"

    return None


def detect_page_sections(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    detected_pages: List[Dict[str, Any]] = []
    current_topic = "front_matter"
    current_subtopic = "front_matter"
    total_pages = len(pages)

    for page in pages:
        page_num = page["page"]
        text = page["text"]
        lines = _page_preview_lines(text)
        candidates = _extract_candidate_headers(lines)

        page_topic = current_topic
        page_subtopic = current_subtopic

        # Front matter / abstract explicit handling.
        if page_num == 1:
            page_topic = "front_matter"
            page_subtopic = "front_matter"
            current_topic = page_topic
            current_subtopic = page_subtopic
        elif any(_normalize_line(line) == "abstract" for _, line in candidates[:5]):
            page_topic = "abstract"
            page_subtopic = "abstract"
            current_topic = page_topic
            current_subtopic = page_subtopic
        else:
            matched_topic = None
            matched_subtopic = None

            for idx, line in candidates:
                topic = _topic_from_header_text(line)
                if not topic:
                    continue
                level = _header_level(line)

                # Major numbered top-level headers or strong plain headers switch section.
                if level == 1 or _looks_like_plain_header(line):
                    matched_topic = topic
                    matched_subtopic = topic
                    break

                # Lower-level numbered headers become subsection markers only.
                if level and level >= 2 and matched_subtopic is None:
                    matched_subtopic = topic

            if matched_topic:
                page_topic = matched_topic
                current_topic = matched_topic
                page_subtopic = matched_subtopic or matched_topic
                current_subtopic = page_subtopic
            else:
                # References fallback near end only.
                joined = " ".join(lines[:20]).lower()
                if page_num >= total_pages - 25 and (
                    "references" in joined[:300] or "bibliography" in joined[:300]
                ):
                    page_topic = "references"
                    page_subtopic = "references"
                    current_topic = page_topic
                    current_subtopic = page_subtopic
                elif current_topic == "abstract":
                    # Abstract must not leak past early pages.
                    page_topic = "introduction"
                    page_subtopic = current_subtopic if current_subtopic != "abstract" else "introduction"
                    current_topic = page_topic
                    current_subtopic = page_subtopic
                else:
                    page_topic = current_topic
                    page_subtopic = current_subtopic

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
    if any(_normalize_line(line) == "abstract" for line in lines[:5]):
        return "abstract"
    for line in lines[:20]:
        topic = _topic_from_header_text(line)
        if topic:
            return CANONICAL_SECTION_MAP.get(topic, "body")
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
