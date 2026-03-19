"""
Domain-specific query expansion for factor investing research papers.

The goal is not to maximize generic semantic similarity, but to align user
intent with the kinds of finance-language phrases that appear in factor
investing papers.
"""
from typing import Dict, List

from app.retrieval.query_router import classify_query

QUERY_TYPE_EXPANSIONS: Dict[str, List[str]] = {
    "contribution": [
        "main contribution", "paper proposes", "paper introduces", "new factor", "new anomaly",
        "return predictor", "cross-sectional return predictability", "investment signal"
    ],
    "factor_definition": [
        "factor construction", "signal construction", "predictor variable", "characteristic definition",
        "sorting variable", "firm characteristic", "anomaly definition", "portfolio sort signal"
    ],
    "data_sample": [
        "dataset", "sample period", "asset universe", "equity universe", "crsp", "compustat",
        "wrds", "nyse amex nasdaq", "monthly returns", "daily returns", "firm-level data"
    ],
    "methodology_portfolio": [
        "portfolio construction", "sorting procedure", "decile sorting", "quintile sorting",
        "long-short portfolio", "value-weighted", "equal-weighted", "rebalancing frequency",
        "empirical design", "double sort"
    ],
    "benchmark": [
        "benchmark model", "baseline model", "fama-french", "capm", "carhart", "market benchmark",
        "existing factors", "compared against", "alpha relative to benchmark"
    ],
    "performance": [
        "alpha", "excess return", "average return", "sharpe ratio", "t-statistic",
        "risk-adjusted return", "spread return", "factor premium", "performance results"
    ],
    "robustness": [
        "robustness check", "transaction costs", "turnover", "subsample analysis",
        "out-of-sample", "international evidence", "alternative specification", "additional tests"
    ],
    "limitations": [
        "limitations", "caveats", "discussion", "future research", "boundary conditions",
        "weaker performance", "implementation frictions"
    ],
    "generic": [
        "factor investing", "asset pricing", "equity returns", "cross-sectional returns"
    ],
}


def expand_query(query: str) -> List[str]:
    """
    Expand a query into a small set of domain-aligned search rewrites.

    The first item is always the original query for backward compatibility.
    """
    query_type = classify_query(query)
    expansions = [query]

    # Add a finance-domain bag-of-terms expansion for the inferred intent.
    domain_terms = QUERY_TYPE_EXPANSIONS.get(query_type, [])
    if domain_terms:
        expansions.append(f"{query} {' '.join(domain_terms[:4])}")
        expansions.append(" ".join(domain_terms))

    # Add some cross-intent support where papers naturally spread answers across
    # sections. For example, performance questions may have summaries in the
    # abstract and details in results.
    if query_type == "performance":
        expansions.append("results alpha sharpe ratio excess return t-statistic")
    elif query_type == "factor_definition":
        expansions.append("methodology factor construction signal definition sorting variable")
    elif query_type == "data_sample":
        expansions.append("data sample period dataset crsp compustat asset universe")
    elif query_type == "methodology_portfolio":
        expansions.append("portfolio construction decile quintile long-short rebalancing")
    elif query_type == "benchmark":
        expansions.append("benchmark fama-french capm baseline comparison alpha")
    elif query_type == "robustness":
        expansions.append("robustness transaction costs turnover subsample out-of-sample")
    elif query_type == "limitations":
        expansions.append("limitations caveats discussion future research")
    elif query_type == "contribution":
        expansions.append("abstract introduction main contribution proposed factor anomaly")

    # Remove duplicates while preserving order.
    seen = set()
    unique = []
    for item in expansions:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(item)

    return unique


def rewrite_for_abstract(query: str) -> str:
    """
    Backward-compatible helper.

    For factor-investing papers, abstract rewrites are now driven by query type.
    """
    query_type = classify_query(query)

    if query_type == "contribution":
        return "abstract main contribution proposed factor anomaly"
    if query_type == "performance":
        return "abstract performance alpha sharpe ratio results"
    if query_type == "data_sample":
        return "abstract data sample period dataset"

    return query
