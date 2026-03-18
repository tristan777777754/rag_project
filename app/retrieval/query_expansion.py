"""
Query expansion and rewriting for better retrieval.
Expands natural questions to better match academic paper content.
"""

# Query expansion templates for common question types
QUERY_EXPANSIONS = {
    # What does this paper introduce/present - EXPANDED for various academic writing styles
    "introduce": ["paper introduces", "this paper", "we present", "we propose", "novel", 
                  "approach", "framework", "method", "here we show", "we demonstrate",
                  "we report", "this study", "we examine", "we investigate"],
    # Problem being solved
    "problem": ["problem", "challenge", "limitations", "computational", "expensive", 
                "prohibitive", "intractable"],
    # Abstract summarization
    "abstract": ["abstract", "paper introduces", "we present", "this paper", "approach",
                 "framework", "method", "results", "here we show"],
    # Method comparison
    "compare": ["compared", "comparison", "versus", "vs", "better", "improvement", 
                "faster", "speedup", "accuracy"],
    # Technical terms
    "pde": ["partial differential equation", "PDE", "deterministic", "stochastic", 
            "differential equation", "Feynman-Kac"],
    # Results/accuracy
    "accuracy": ["accuracy", "cents", "pricing accuracy", "error", "benchmark", 
                 "Monte Carlo", "compared"],
    # Method details
    "method": ["method", "approach", "framework", "algorithm", "neural network", 
               "FINN", "Finance-Informed"],
}

def expand_query(query: str) -> list[str]:
    """
    Expand a query into multiple variations for better retrieval.
    Returns a list of query strings to search.
    """
    query_lower = query.lower()
    expanded = [query]  # Always include original
    
    # Check for question patterns and expand
    if any(kw in query_lower for kw in ["introduce", "present", "about", "what is"]) and "paper" in query_lower:
        # Question about paper introduction - expand with academic terms
        expanded.append("paper introduces framework approach method")
        expanded.append("abstract this paper presents")
        expanded.append("we propose novel approach")
    
    if "problem" in query_lower or "solve" in query_lower:
        expanded.append("problem challenge computational limitations")
        expanded.append("addresses the problem of")
    
    if "abstract" in query_lower or "summarize" in query_lower:
        expanded.append("abstract paper introduces approach")
    
    if "compare" in query_lower or "versus" in query_lower or "better" in query_lower:
        expanded.append("compared to Monte Carlo speedup faster")
        expanded.append("comparison benchmark results")
    
    if "pde" in query_lower or "differential equation" in query_lower:
        expanded.append("partial differential equation PDE Feynman-Kac")
        expanded.append("stochastic deterministic PDE formulation")
    
    if "accuracy" in query_lower or "achieve" in query_lower:
        expanded.append("accuracy error cents benchmark")
        expanded.append("pricing accuracy within cents")
    
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for q in expanded:
        if q.lower() not in seen:
            seen.add(q.lower())
            unique.append(q)
    
    return unique


def rewrite_for_abstract(query: str) -> str:
    """
    Rewrite a question to better match abstract content.
    """
    query_lower = query.lower()
    
    # Pattern: "what does this paper introduce" -> match abstract content
    if "introduce" in query_lower or "present" in query_lower or "about" in query_lower:
        return "paper introduces approach framework method presents"
    
    # Pattern: "what problem" -> match problem statements
    if "problem" in query_lower and ("what" in query_lower or "solve" in query_lower):
        return "problem challenge computational stochastic numerical methods"
    
    # Pattern: "how does X compare" -> match comparison content
    if "compare" in query_lower or "versus" in query_lower or "faster" in query_lower:
        return "compared Monte Carlo speedup faster benchmark performance"
    
    return query
