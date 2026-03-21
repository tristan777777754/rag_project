"""Hybrid document-structure parsing for finance papers with layout-aware headers."""
import re
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple, Set

CANONICAL_SECTION_MAP = {
    "front_matter": "front_matter",
    "abstract": "abstract",
    "introduction": "introduction",
    "related_literature": "introduction",
    "methodology": "methodology",
    "data_sample": "methodology",
    "performance": "results",
    "robustness": "results",
    "discussion": "body",
    "conclusion": "conclusion",
    "references": "references",
    "appendix": "body",
    "figures_tables": "body",
    "acknowledgments": "acknowledgments",
    "body": "body",
}

PLAIN_HEADER_WHITELIST = {
    "abstract", "introduction", "related literature", "methodology", "data", "conclusion", "references",
    "bibliography", "appendix", "simulation procedure", "tables", "figures", "robustness", "methods",
    "results", "summary", "questions", "mechanisms", "factor models", "intuition", "subsequent work",
    "structural estimation", "quantitative theories", "implications", "other asset classes",
    "empirical properties", "a review", "new factors", "comparison on conceptual grounds",
    "comparison on empirical grounds", "pricing errors and tests of overall performance", "factor loadings",
    "out-of-sample alphas", "the ff five-factor model", "motivation for the q-factor model",
    "motivation for the ff five-factor model from valuation theory", "four concerns on the ff (2014a) motivation",
    "all-but-micro breakpoints and equal-weighted returns", "alternative factor constructions",
    "regressions with alternative factors"
}


def _normalize_line(line: str) -> str:
    line = line.strip().lower()
    line = line.replace("ﬁ", "fi").replace("ﬂ", "fl")
    line = line.replace("–", "-").replace("—", "-")
    line = re.sub(r"\s+", " ", line)
    return line.strip(" .:-")


def _topic_slug(text: str) -> str:
    t = _normalize_line(text)
    t = re.sub(r"[^a-z0-9]+", "_", t)
    return t.strip("_")


def _page_preview_lines(text: str, limit: int = 60) -> List[str]:
    return [line.strip() for line in text.split("\n")[:limit] if line.strip()]


def _clean_header_text(line: str) -> str:
    normalized = _normalize_line(line)
    normalized = re.sub(r"^(section\s+)?(\d+(\.\d+)*|[ivxlcdm]+|[a-z])\s*[\.)-]?\s+", "", normalized)
    return normalized.strip()


def _header_level(line: str) -> Optional[int]:
    normalized = _normalize_line(line)
    if re.match(r"^\d+\.\d+\.\d+\s*[\.)-]?\s+[a-z]", normalized):
        return 3
    if re.match(r"^\d+\.\d+\s*[\.)-]?\s+[a-z]", normalized):
        return 2
    if re.match(r"^\d+\s*[\.)-]?\s+[a-z]", normalized):
        return 1
    if re.match(r"^[ivxlcdm]+\s*[\.)-]?\s+[a-z]", normalized):
        return 1
    return None


def _is_numbered_header(line: str) -> bool:
    normalized = _normalize_line(line)
    return re.match(r"^(\d+(\.\d+)*|[ivxlcdm]+)\s*[\.)-]?\s+[a-z]", normalized) is not None


def _looks_like_plain_header(line: str) -> bool:
    raw = line.strip()
    normalized = _normalize_line(raw)
    if not normalized or len(normalized) > 80:
        return False
    if re.search(r"[!?;,]", raw):
        return False
    if raw.endswith('.'):
        return False
    if len(raw.split()) > 8:
        return False
    return normalized in PLAIN_HEADER_WHITELIST


def _boilerplate_top_lines(pages: List[Dict[str, Any]]) -> Set[str]:
    ctr = Counter()
    for page in pages:
        source_lines = page.get("lines_meta") or [{"text": ln} for ln in _page_preview_lines(page["text"], 10)]
        for item in source_lines[:8]:
            norm = _normalize_line(item["text"])
            if norm:
                ctr[norm] += 1
    return {line for line, count in ctr.items() if count >= 3 or re.fullmatch(r"\d+", line)}


def _is_layout_header(item: Dict[str, Any], body_size: float) -> bool:
    text = item["text"].strip()
    norm = _normalize_line(text)
    if not norm:
        return False
    if len(text) > 100 or len(text.split()) > 12:
        return False
    if text.endswith('.'):
        return False
    if re.search(r"[!?;=\[\]{}<>]|\b(t-value|p-value)\b", text, re.I):
        return False
    alpha_count = sum(ch.isalpha() for ch in text)
    digit_count = sum(ch.isdigit() for ch in text)
    if alpha_count < 3 or digit_count > alpha_count:
        return False
    size_boost = item.get("size", 0) >= body_size + 0.8 if body_size else False
    style_boost = item.get("flags", 0) >= 16 or item.get("is_upper", False)
    numbered = _header_level(text) is not None
    whitelisted = _looks_like_plain_header(text)
    return numbered or whitelisted or size_boost or style_boost


def _extract_candidate_headers(page: Dict[str, Any], boilerplate: Set[str]) -> List[Tuple[float, str, int]]:
    candidates = []
    lines_meta = page.get("lines_meta") or []
    body_size = page.get("body_font_size", 0)

    if lines_meta:
        for idx, item in enumerate(lines_meta):
            text = item["text"].strip()
            norm = _normalize_line(text)
            if not norm or norm in boilerplate:
                continue
            if item["y0"] > 540:
                continue
            if not _is_layout_header(item, body_size):
                continue
            level = _header_level(text) or 1
            score = 0.0
            if _header_level(text) is not None:
                score += 3.0
            if _is_numbered_header(text):
                score += 2.5
            if _looks_like_plain_header(text):
                score += 2.0
            if item.get("size", 0) >= body_size + 0.8:
                score += 1.5
            if item.get("flags", 0) >= 16:
                score += 1.5
            if item.get("is_upper", False):
                score += 0.8
            score += max(0, 1.2 - item["y0"] / 400.0)
            candidates.append((score, text, level))

            # Combine split headers like "2" + "New Factors"
            if re.fullmatch(r"(\d+(\.\d+)*|[ivxlcdm]+|[a-z])", norm) and idx + 1 < len(lines_meta):
                nxt = lines_meta[idx + 1]
                nxt_text = nxt["text"].strip()
                nxt_norm = _normalize_line(nxt_text)
                if nxt_norm and nxt_norm not in boilerplate and _is_layout_header(nxt, body_size):
                    combined = f"{text} {nxt_text}"
                    clevel = _header_level(combined)
                    if clevel is not None:
                        cscore = score + 2.0
                        if _looks_like_plain_header(nxt_text):
                            cscore += 1.0
                        candidates.append((cscore, combined, clevel))
    else:
        for i, line in enumerate(_page_preview_lines(page["text"], 20)):
            norm = _normalize_line(line)
            if not norm or norm in boilerplate:
                continue
            if _header_level(line) is not None or _looks_like_plain_header(line):
                candidates.append((1.0, line, _header_level(line) or 1))

    candidates.sort(key=lambda x: (-x[0], x[2]))
    return candidates


def _broad_topic_from_text(header_text: str, current_broad: str = "body") -> str:
    h = _clean_header_text(header_text)

    if h == "abstract": return "abstract"
    if h in {"introduction", "summary", "questions", "overview", "related literature", "a review"}: return "introduction"
    if h in {"conclusion", "conclusions", "concluding remarks"}: return "conclusion"
    if h in {"references", "bibliography"}: return "references"
    if h in {"appendix", "simulation procedure"}: return "appendix"
    if h in {"tables", "figures"}: return "figures_tables"
    if h in {"empirical properties", "out-of-sample alphas", "factor loadings", "pricing errors and tests of overall performance", "all-but-micro breakpoints and equal-weighted returns", "comparison in empirical performance", "the playing field", "alternative factor constructions", "regressions with alternative factors"}: return "performance"
    if h in {"comparison on conceptual grounds", "four concerns on the ff (2014a) motivation", "implications", "other asset classes"}: return "discussion"

    if any(x in h for x in ["method", "methodology", "factor construction", "bayesian procedure", "gibbs sampling", "factor model", "structural estimation", "quantitative theor", "new factors"]):
        return "methodology"
    if any(x in h for x in ["data", "sample", "dataset", "empirical setting"]):
        return "data_sample"
    if any(x in h for x in ["results", "empirical propert", "performance", "comparison on empirical grounds", "all-but-micro breakpoints", "out-of-sample alphas", "factor loadings", "pricing errors and tests of overall performance", "transaction costs and ", "average transaction costs", "dynamics of transaction costs"]):
        return "performance"
    if any(x in h for x in ["robustness", "additional tests", "other conditioning", "tail risk", "vix", "lot model", "lesmond", "cost-mitigating trading strategies"]):
        return "robustness"
    if any(x in h for x in ["comparison on conceptual grounds", "implications", "other asset classes", "emh counterrevolution", "complementarity", "how i defend fama", "security analysis", "rational expectations economics", "challenges", "four concerns"]):
        return "discussion"

    if current_broad in {"introduction", "discussion"} and h in {"mechanisms", "methods", "factor models", "intuition", "subsequent work"}:
        return current_broad

    return current_broad if current_broad != "front_matter" else "body"


def _references_like(page: Dict[str, Any]) -> bool:
    text = page["text"]
    lines = [ln.strip() for ln in text.splitlines()[:50] if ln.strip()]
    joined = " ".join(lines).lower()
    if any(_normalize_line(ln) in {"references", "bibliography"} for ln in lines[:10]):
        return True
    patterns = [r"\(\d{4}\)", r"journal of", r"review of", r"econometrica", r"doi"]
    return sum(1 for p in patterns if re.search(p, joined)) >= 3


def detect_page_sections(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    detected_pages = []
    boilerplate = _boilerplate_top_lines(pages)
    current_broad = current_topic = current_subtopic = "front_matter"
    total_pages = len(pages)

    for page in pages:
        page_num = page["page"]
        lines = _page_preview_lines(page["text"], 20)
        candidates = _extract_candidate_headers(page, boilerplate)
        broad, topic, subtopic = current_broad, current_topic, current_subtopic

        if page_num == 1:
            broad = topic = subtopic = "front_matter"
        elif any(_normalize_line(ln) == "abstract" for ln in lines[:12]) or any(_clean_header_text(text) == "abstract" for _, text, _ in candidates[:10]):
            broad = topic = subtopic = "abstract"
        else:
            matched_major = None
            matched_sub = None
            numbered_level1 = [c for c in candidates if c[2] == 1 and _is_numbered_header(c[1])]
            scan_candidates = numbered_level1 + [c for c in candidates if c not in numbered_level1]
            for score, text, level in scan_candidates:
                cleaned = _clean_header_text(text)
                if not cleaned:
                    continue
                slug = _topic_slug(cleaned)
                broad_guess = _broad_topic_from_text(text, current_broad)
                strong_major = level == 1 or score >= 4.5
                if strong_major:
                    matched_major = (slug, broad_guess)
                    matched_sub = slug
                    break
                if matched_sub is None:
                    matched_sub = slug
                    if broad_guess != current_broad and broad_guess not in {"body", "front_matter"} and score >= 2.5:
                        matched_major = (slug, broad_guess)
                        break

            if matched_major:
                topic, broad = matched_major
                subtopic = matched_sub or topic
            else:
                if current_broad == "abstract":
                    broad = topic = subtopic = "introduction"
                else:
                    broad = current_broad
                    topic = current_topic
                    subtopic = matched_sub or current_subtopic
                if page_num >= max(8, total_pages - 8) and _references_like(page):
                    broad = topic = subtopic = "references"

        enriched = dict(page)
        enriched["section"] = CANONICAL_SECTION_MAP.get(broad, "body")
        enriched["section_topic"] = topic
        enriched["subsection_topic"] = subtopic
        enriched["broad_topic"] = broad
        detected_pages.append(enriched)
        current_broad, current_topic, current_subtopic = broad, topic, subtopic

    return detected_pages


def detect_section(text: str, page_num: int) -> str:
    if page_num == 1:
        return "front_matter"
    lines = _page_preview_lines(text)
    if any(_normalize_line(line) == "abstract" for line in lines[:8]):
        return "abstract"
    for line in lines[:20]:
        broad = _broad_topic_from_text(line, "body")
        if broad != "body":
            return CANONICAL_SECTION_MAP.get(broad, "body")
    return "body"


def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def split_pages_into_chunks_semantic(pages: List[Dict[str, Any]], max_chunk_size: int = 800, min_chunk_size: int = 200, overlap_sentences: int = 1) -> List[Dict[str, Any]]:
    chunks = []
    pages_with_sections = detect_page_sections(pages)
    for page in pages_with_sections:
        text = page["text"]
        page_num = page["page"]
        section = page.get("section", "body")
        section_topic = page.get("section_topic", section)
        subsection_topic = page.get("subsection_topic", section_topic)
        broad_topic = page.get("broad_topic", section)
        if len(text.strip()) < min_chunk_size:
            if page_num <= 2 and len(text.strip()) > 50:
                chunks.append({"text": text.strip(), "page": page_num, "section": section, "section_topic": section_topic, "subsection_topic": subsection_topic, "broad_topic": broad_topic, "position": "start" if page_num <= 2 else "body"})
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
                    chunks.append({"text": chunk_text, "page": page_num, "section": section, "section_topic": section_topic, "subsection_topic": subsection_topic, "broad_topic": broad_topic, "position": "start" if page_num <= 2 and not chunks else "body"})
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
                chunks.append({"text": chunk_text, "page": page_num, "section": section, "section_topic": section_topic, "subsection_topic": subsection_topic, "broad_topic": broad_topic, "position": "start" if page_num <= 2 and not chunks else "body"})
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
    return split_pages_into_chunks_semantic(pages, max_chunk_size=chunk_size, overlap_sentences=1 if overlap > 0 else 0)
