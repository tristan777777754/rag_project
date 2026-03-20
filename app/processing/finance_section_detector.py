import re
from typing import List, Dict, Any


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _first_lines(text: str, n: int = 12) -> List[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[:n]


def _is_front_matter(page_num: int, text: str) -> bool:
    if page_num > 2:
        return False

    t = text.lower()
    first_2000 = t[:2000]

    signals = [
        "electronic copy available",
        "corresponding author",
        "university",
        "school of economics",
        "department of",
        "jel classification",
        "keywords",
        "working paper",
        "national bureau of economic research",
        "nber working paper",
        "@",
    ]

    score = sum(1 for s in signals if s in first_2000)

    if page_num == 1 and score >= 2:
        return True

    return False


def _detect_explicit_header(text: str) -> str | None:
    """
    Only use strong header signals.
    Do NOT classify robustness/results/data from a casual mention in body text.
    """
    for raw_line in _first_lines(text, 12):
        line = _normalize(raw_line)

        # remove numbering like:
        # 1 Introduction
        # 1. Introduction
        # 2.3 Data
        line_no_num = re.sub(r"^(section\s+)?(\d+(\.\d+)*|[ivxlcdm]+)\.?\s+", "", line).strip()

        # strong exact-ish header matches
        if line_no_num in {"abstract", "summary"}:
            return "abstract"

        if line_no_num in {"introduction", "overview"}:
            return "introduction"

        if line_no_num in {"conclusion", "conclusions", "concluding remarks"}:
            return "conclusion"

        if line_no_num in {"references", "bibliography"}:
            return "references"

        if line_no_num in {"appendix", "internet appendix", "online appendix"}:
            return "appendix"

        # finance-specific strong headers
        if (
            "data" == line_no_num
            or "sample" == line_no_num
            or "data and sample" in line_no_num
            or "variables and data" in line_no_num
            or "data description" in line_no_num
        ):
            return "data_sample"

        if (
            "portfolio construction" in line_no_num
            or "empirical methodology" in line_no_num
            or "methodology" == line_no_num
            or "research design" in line_no_num
            or "sorting procedure" in line_no_num
            or "factor model" in line_no_num
            or "the q-factor model" in line_no_num
            or "q-factor model" in line_no_num
        ):
            return "methodology_portfolio"

        if (
            line_no_num == "results"
            or "empirical results" in line_no_num
            or "main results" in line_no_num
            or line_no_num == "performance"
        ):
            return "performance"

        # IMPORTANT: robustness only from strong header
        if (
            line_no_num == "robustness"
            or "robustness checks" in line_no_num
            or "additional tests" in line_no_num
        ):
            return "robustness"

    return None


def _is_abstract_like(page_num: int, text: str) -> bool:
    if page_num > 2:
        return False

    t = text.lower()
    first_1800 = t[:1800]

    if "abstract" in first_1800:
        return True

    # abstract-like summary cues
    cues = [
        "this paper",
        "we study",
        "we examine",
        "we estimate",
        "we propose",
        "we find",
        "we show",
        "using data",
        "we compute",
        "we document",
    ]
    score = sum(1 for c in cues if c in first_1800)

    # avoid classifying obvious title/cover page as abstract
    if _is_front_matter(page_num, text):
        return False

    return score >= 3


def _is_intro_like(page_num: int, text: str) -> bool:
    if page_num > 4:
        return False

    t = text.lower()
    first_1800 = t[:1800]

    cues = [
        "the literature",
        "this paper",
        "we study",
        "we examine",
        "we contribute",
        "our contribution",
        "motivated by",
    ]
    score = sum(1 for c in cues if c in first_1800)

    return score >= 2


def _is_references_like(page_num: int, total_pages: int, text: str) -> bool:
    t = text.lower()

    if "references" in t[:500] or "bibliography" in t[:500]:
        return True

    if page_num >= total_pages - 2:
        patterns = [
            r"\(\d{4}\)",
            r"et al\.",
            r"journal",
            r"vol\.",
            r"pp\.",
            r"doi",
        ]
        score = sum(1 for p in patterns if re.search(p, t))
        return score >= 3

    return False


def build_page_section_map(pages: List[Dict[str, Any]]) -> Dict[int, str]:
    page_map: Dict[int, str] = {}
    total_pages = len(pages)
    current_section = "body"

    for page in pages:
        page_num = page["page"]
        text = page["text"]

        # 1. strongest rule: front matter
        if _is_front_matter(page_num, text):
            current_section = "front_matter"
            page_map[page_num] = current_section
            continue

        # 2. strongest rule: explicit header
        explicit = _detect_explicit_header(text)
        if explicit:
            current_section = explicit
            page_map[page_num] = current_section
            continue

        # 3. early-page conservative heuristics
        if page_num <= 2 and _is_abstract_like(page_num, text):
            current_section = "abstract"
            page_map[page_num] = current_section
            continue

        if page_num <= 4 and _is_intro_like(page_num, text):
            current_section = "introduction"
            page_map[page_num] = current_section
            continue

        # 4. references near the end
        if _is_references_like(page_num, total_pages, text):
            current_section = "references"
            page_map[page_num] = current_section
            continue

        # 5. otherwise inherit previous section
        page_map[page_num] = current_section

    return page_map