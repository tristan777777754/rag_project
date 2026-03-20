import re
from typing import List, Dict, Any, Optional


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _first_lines(text: str, n: int = 20) -> List[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[:n]


def _is_front_matter(page_num: int, text: str) -> bool:
    """
    Detect title / cover / author page.
    Only for very early pages.
    """
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
    return page_num == 1 and score >= 2


def _strip_number_prefix(line: str) -> str:
    """
    Examples:
    1 Introduction
    1. Introduction
    2.1 Data
    I. Introduction
    """
    line = _normalize(line)
    line = re.sub(r"^(section\s+)?(\d+(\.\d+)*|[ivxlcdm]+)\.?\s+", "", line).strip()
    return line


def _looks_like_header(line: str) -> bool:
    """
    Strong header-like pattern only.
    Avoid classifying normal body sentences as headers.
    """
    raw = line.strip()
    if not raw:
        return False

    if len(raw) > 120:
        return False

    line_norm = _normalize(raw)

    # numbered headers
    if re.match(r"^(\d+(\.\d+)*|[ivxlcdm]+)\.?\s+[a-z]", line_norm):
        return True

    # plain short section titles
    if line_norm in {
        "abstract",
        "introduction",
        "related literature",
        "methodology",
        "data",
        "results",
        "conclusion",
        "references",
        "appendix",
        "tables",
        "figures",
    }:
        return True

    # headers with colon
    if ":" in raw and len(raw.split()) <= 14:
        return True

    return False


def _map_header_to_section(line: str) -> Optional[str]:
    """
    Header-driven mapping only.
    """
    line_norm = _strip_number_prefix(line)

    # exact / strong academic headers
    if line_norm in {"abstract", "summary"}:
        return "abstract"

    if line_norm in {"introduction", "related literature", "overview"}:
        return "introduction"

    if line_norm in {"conclusion", "conclusions", "concluding remarks"}:
        return "conclusion"

    if line_norm in {"references", "bibliography"}:
        return "references"

    if line_norm in {
        "appendix",
        "internet appendix",
        "online appendix",
        "simulation procedure",
        "tables",
        "figures",
    }:
        return "appendix"

    # data-like
    if (
        line_norm == "data"
        or line_norm == "sample"
        or "data and sample" in line_norm
        or "variables and data" in line_norm
        or "data description" in line_norm
    ):
        return "data_sample"

    # methodology-like
    if (
        line_norm == "methodology"
        or "portfolio construction" in line_norm
        or "empirical methodology" in line_norm
        or "research design" in line_norm
        or "sorting procedure" in line_norm
        or "factor model" in line_norm
        or "the q-factor model" in line_norm
        or "q-factor model" in line_norm
        or "measuring the effective bid-ask spread" in line_norm
        or "the model of roll" in line_norm
        or "the bayesian procedure" in line_norm
        or "hasbouck model with funding liquidity" in line_norm
        or "hasbrouck model with funding liquidity" in line_norm
        or "estimation by gibbs sampling" in line_norm
        or "bayes factor" in line_norm
    ):
        return "methodology_portfolio"

    # performance / results-like
    if (
        line_norm == "results"
        or "empirical results" in line_norm
        or "main results" in line_norm
        or line_norm == "performance"
        or "transaction costs with funding liquidity" in line_norm
        or "average transaction costs" in line_norm
        or "all stocks" in line_norm
        or "anomaly portfolios" in line_norm
        or "the dynamics of transaction costs" in line_norm
        or "transaction costs and firm size" in line_norm
        or "transaction costs and volatility" in line_norm
        or "transaction costs and momentum" in line_norm
        or "transaction costs and flight to quality" in line_norm
        or "after-trading-cost performance of anomalies" in line_norm
        or "gross returns and net returns of anomalies" in line_norm
        or "performance of long-short strategies" in line_norm
        or "cost-mitigating trading strategies" in line_norm
    ):
        return "performance"

    # robustness-like
    if (
        line_norm == "robustness"
        or "robustness checks" in line_norm
        or "additional tests" in line_norm
        or line_norm.startswith("robustness:")
    ):
        return "robustness"

    return None


def _detect_explicit_header(text: str) -> Optional[str]:
    """
    Only trust strong header lines near the top of page.
    """
    for raw_line in _first_lines(text, 20):
        if not _looks_like_header(raw_line):
            continue

        mapped = _map_header_to_section(raw_line)
        if mapped:
            return mapped

    return None


def _is_abstract_like(page_num: int, text: str) -> bool:
    """
    Very limited fallback:
    only allow abstract on early pages.
    """
    if page_num > 2:
        return False

    if _is_front_matter(page_num, text):
        return False

    t = text.lower()
    first_1800 = t[:1800]

    first_300 = first_1800[:300]

    if "abstract" in first_300:
        return True

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

    stop_signals = [
        "1 introduction",
        "introduction",
        "section 2",
        "section 3",
        "section 4",
        "section 5",
        "section 6",
        "section 7",
        "the rest of the paper is structured as follows",
    ]
    if any(s in first_1800 for s in stop_signals):
        return False

    return score >= 3


def _is_references_like(page_num: int, total_pages: int, text: str) -> bool:
    t = text.lower()

    if "references" in t[:500] or "bibliography" in t[:500]:
        return True

    if page_num >= total_pages - 3:
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
    """
    Header-driven section detector.

    Core logic:
    1. front matter
    2. explicit header
    3. very limited abstract fallback
    4. references near end
    5. otherwise inherit previous section

    This is intentionally NOT keyword-heavy.
    """
    page_map: Dict[int, str] = {}
    total_pages = len(pages)
    current_section = "body"

    for page in pages:
        page_num = page["page"]
        text = page["text"]

        # 1. cover / title page
        if _is_front_matter(page_num, text):
            current_section = "front_matter"
            page_map[page_num] = current_section
            continue

        # 2. explicit header = strongest signal
        explicit = _detect_explicit_header(text)
        if explicit:
            current_section = explicit
            page_map[page_num] = current_section
            continue

        # 3. limited abstract fallback for early pages only
        if _is_abstract_like(page_num, text):
            current_section = "abstract"
            page_map[page_num] = current_section
            continue

        # 4. references near the end
        if _is_references_like(page_num, total_pages, text):
            current_section = "references"
            page_map[page_num] = current_section
            continue

        # 5. special fix:
        # abstract should not keep leaking forever
        if current_section == "abstract":
            current_section = "introduction"

        # 6. inherit previous section
        page_map[page_num] = current_section

    return page_map