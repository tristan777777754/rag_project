"""
Query Router for factor investing research papers.

This module moves routing from generic academic-paper intent labels to
factor-investing-specific query types. The router is deterministic so that
benchmark results remain stable and explainable.
"""
from typing import Dict, List

# Canonical section labels currently produced by text_splitter.py.
CANONICAL_SECTIONS = {
    "abstract",
    "introduction",
    "methodology",
    "results",
    "conclusion",
    "references",
    "acknowledgments",
    "body",
}

# Query-type-specific target section aliases. These are human-friendly labels
# used by the benchmark/prompting layer. They are later mapped back to the
# canonical section labels available in the current pipeline.
QUERY_TYPE_TARGETS: Dict[str, List[str]] = {
    "contribution": ["abstract", "introduction", "contribution", "summary"],
    "factor_definition": ["methodology", "introduction", "factor definition", "signal construction"],
    "data_sample": ["data", "sample", "dataset", "empirical setting", "methodology"],
    "methodology_portfolio": ["methodology", "portfolio construction", "sorting procedure", "empirical setting"],
    "benchmark": ["results", "methodology", "benchmark", "comparison"],
    "performance": ["results", "empirical results", "tables", "abstract"],
    "robustness": ["robustness", "appendix", "additional tests", "results", "conclusion"],
    "limitations": ["conclusion", "discussion", "limitations", "abstract"],
    "generic": ["abstract", "introduction", "methodology", "results", "conclusion", "body"],
}

# Map flexible section aliases to canonical section labels in the existing
# corpus. This is what makes section-aware filtering possible without changing
# the chunker first.
SECTION_ALIAS_TO_CANONICAL: Dict[str, List[str]] = {
    "abstract": ["abstract"],
    "introduction": ["introduction", "abstract"],
    "contribution": ["abstract", "introduction"],
    "summary": ["abstract", "introduction", "conclusion"],
    "factor definition": ["methodology", "introduction", "body"],
    "signal construction": ["methodology", "body"],
    "data": ["methodology", "body", "results"],
    "sample": ["methodology", "body", "results"],
    "dataset": ["methodology", "body", "results"],
    "empirical setting": ["methodology", "body", "results"],
    "portfolio construction": ["methodology", "body", "results"],
    "sorting procedure": ["methodology", "body"],
    "benchmark": ["results", "methodology", "body"],
    "comparison": ["results", "methodology"],
    "empirical results": ["results", "abstract", "conclusion"],
    "tables": ["results", "body"],
    "robustness": ["results", "conclusion", "body"],
    "appendix": ["results", "body", "conclusion"],
    "additional tests": ["results", "body", "conclusion"],
    "limitations": ["conclusion", "abstract", "body"],
    "discussion": ["conclusion", "body"],
    "methodology": ["methodology", "body"],
    "results": ["results", "abstract", "conclusion"],
    "conclusion": ["conclusion", "abstract"],
    "body": ["body"],
}

QUERY_TYPE_KEYWORDS = {
    "factor_definition": [
        "factor", "signal", "anomaly", "constructed", "construction", "define", "definition",
        "characteristic", "predictor", "measure", "measured", "computed", "sorted on"
    ],
    "data_sample": [
        "dataset", "data", "sample", "sample period", "time period", "observation period",
        "universe", "crsp", "compustat", "wrds", "stocks", "firms", "market", "monthly", "daily"
    ],
    "methodology_portfolio": [
        "portfolio", "portfolios", "long-short", "long short", "decile", "quintile", "sort", "sorting",
        "rebalance", "rebalancing", "weight", "weighted", "value-weighted", "equal-weighted",
        "methodology", "empirical design", "formation"
    ],
    "benchmark": [
        "benchmark", "baseline", "compare", "comparison", "fama-french", "capm", "market factor",
        "versus", "vs", "outperform", "relative to"
    ],
    "performance": [
        "performance", "alpha", "return", "returns", "sharpe", "t-stat", "t statistic", "drawdown",
        "excess return", "profitability", "spread", "premium", "risk-adjusted"
    ],
    "robustness": [
        "robust", "robustness", "transaction cost", "transaction costs", "turnover", "subsample",
        "out-of-sample", "international", "additional test", "sensitivity", "stability"
    ],
    "limitations": [
        "limitation", "limitations", "weakness", "caveat", "future work", "fails", "underperform",
        "does not", "may not", "constraint"
    ],
    "contribution": [
        "contribution", "contribute", "introduce", "propose", "paper", "study", "main idea",
        "what does", "what is this paper", "summary", "overview", "anomaly"
    ],
}

# Early pages are still useful as a fallback, but page priors are much lighter
# than before because section-aware filtering should do most of the work now.
QUERY_TYPE_PAGE_BOOSTS: Dict[str, Dict[int, float]] = {
    "contribution": {1: 2.5, 2: 1.8, 3: 1.3},
    "factor_definition": {1: 1.3, 2: 1.4, 3: 1.2},
    "data_sample": {2: 1.3, 3: 1.3, 4: 1.1},
    "methodology_portfolio": {2: 1.3, 3: 1.3, 4: 1.2},
    "benchmark": {},
    "performance": {2: 1.4, 3: 1.1},
    "robustness": {},
    "limitations": {1: 1.1},
    "generic": {1: 1.2, 2: 1.1},
}


def _contains_any(query_lower: str, keywords: List[str]) -> bool:
    return any(keyword in query_lower for keyword in keywords)


def classify_query(query: str) -> str:
    """
    Deterministically classify a factor-investing query.

    Returns one of:
    contribution, factor_definition, data_sample, methodology_portfolio,
    benchmark, performance, robustness, limitations, generic
    """
    query_lower = query.lower().strip()

    # More specific intents first to avoid broad keywords such as "what is"
    ordered_types = [
        "robustness",
        "limitations",
        "performance",
        "benchmark",
        "methodology_portfolio",
        "factor_definition",
        "data_sample",
        "contribution",
    ]

    # Contribution-style questions should win when the user asks what the paper
    # proposes or contributes, even if the word "factor" also appears.
    if (
        ("propose" in query_lower or "introduce" in query_lower or "contribution" in query_lower)
        and ("paper" in query_lower or "study" in query_lower)
    ):
        return "contribution"

    for query_type in ordered_types:
        if _contains_any(query_lower, QUERY_TYPE_KEYWORDS[query_type]):
            return query_type

    return "generic"


def get_target_sections(query: str) -> List[str]:
    """Return human-friendly target sections for the query type."""
    query_type = classify_query(query)
    return QUERY_TYPE_TARGETS.get(query_type, QUERY_TYPE_TARGETS["generic"])


def get_query_route(query: str) -> Dict[str, List[str] | str]:
    """
    Route a query to its query type and target sections.

    This is the main structured output used by the retriever.
    """
    query_type = classify_query(query)
    return {
        "query_type": query_type,
        "target_sections": get_target_sections(query),
    }


def map_target_sections_to_canonical(target_sections: List[str], available_sections: List[str] | None = None) -> List[str]:
    """
    Convert flexible section aliases into canonical section labels used in
    chunk metadata.

    If available_sections is provided, keep only sections that exist in the
    current document set to avoid over-filtering on impossible labels.
    """
    canonical = []
    available = set(available_sections or CANONICAL_SECTIONS)

    for label in target_sections:
        label_lower = label.lower()
        candidates = SECTION_ALIAS_TO_CANONICAL.get(label_lower, [label_lower])
        for candidate in candidates:
            if candidate in available and candidate not in canonical:
                canonical.append(candidate)

    # Never route into these sections.
    canonical = [s for s in canonical if s not in {"references", "acknowledgments"}]

    # Safe fallback if nothing matches.
    if not canonical:
        canonical = [s for s in ["abstract", "introduction", "methodology", "results", "conclusion", "body"] if s in available]

    return canonical


def get_page_boost_for_query(query: str) -> Dict[int, float]:
    """Return lightweight page priors for fallback reranking."""
    query_type = classify_query(query)
    return QUERY_TYPE_PAGE_BOOSTS.get(query_type, QUERY_TYPE_PAGE_BOOSTS["generic"])


def get_section_boost_for_query(query: str, available_sections: List[str] | None = None) -> Dict[str, float]:
    """
    Convert routed target sections into soft boosts.

    Even with section filtering, soft boosting remains useful because target
    sections can map to multiple canonical labels.
    """
    available = available_sections or list(CANONICAL_SECTIONS)
    route = get_query_route(query)
    preferred_sections = map_target_sections_to_canonical(route["target_sections"], available)

    base_weights = {section: 0.85 for section in available}
    for section in preferred_sections:
        base_weights[section] = 1.8

    # Keep noisy sections effectively disabled.
    for noisy in ("references", "acknowledgments"):
        if noisy in base_weights:
            base_weights[noisy] = 0.0

    # Fine-tune certain query types where one canonical section should dominate.
    query_type = route["query_type"]
    if query_type == "contribution":
        for sec, weight in {"abstract": 2.4, "introduction": 2.0, "conclusion": 1.2}.items():
            if sec in base_weights:
                base_weights[sec] = weight
    elif query_type == "performance":
        for sec, weight in {"results": 2.4, "abstract": 3.0, "conclusion": 1.3, "body": 0.75}.items():
            if sec in base_weights:
                base_weights[sec] = weight
    elif query_type == "factor_definition":
        for sec, weight in {"methodology": 2.3, "introduction": 1.6, "body": 1.3}.items():
            if sec in base_weights:
                base_weights[sec] = weight
    elif query_type == "data_sample":
        for sec, weight in {"methodology": 2.0, "results": 1.3, "body": 1.3}.items():
            if sec in base_weights:
                base_weights[sec] = weight
    elif query_type == "robustness":
        for sec, weight in {"results": 2.2, "conclusion": 2.8, "body": 1.0, "abstract": 0.2}.items():
            if sec in base_weights:
                base_weights[sec] = weight
    elif query_type == "limitations":
        for sec, weight in {"conclusion": 2.2, "abstract": 1.3, "body": 1.2}.items():
            if sec in base_weights:
                base_weights[sec] = weight

    return base_weights


def should_include_section(query: str, section: str, available_sections: List[str] | None = None) -> bool:
    """
    Determine whether a section should be included in the filtered retrieval
    pool for this query.
    """
    if section in {"acknowledgments", "references"}:
        return False

    route = get_query_route(query)
    allowed = set(map_target_sections_to_canonical(route["target_sections"], available_sections))
    return section in allowed
