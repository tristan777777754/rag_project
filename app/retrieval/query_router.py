"""
Query Router - Dynamically adjusts retrieval strategy based on query type.
Routes queries to appropriate sections with tailored boosting.
"""
from typing import Dict


def classify_query(query: str) -> str:
    """
    Classify the query type based on keywords.
    Returns: 'introduction', 'problem', 'method', 'results', 'comparison', 'generic'
    """
    query_lower = query.lower()
    
    # Introduction/Overview queries
    intro_keywords = ['introduce', 'about', 'overview', 'what is', 'what does', 
                      'purpose', 'goal', 'aim', 'objective', 'main idea']
    if any(kw in query_lower for kw in intro_keywords):
        return 'introduction'
    
    # Problem/Challenge queries
    problem_keywords = ['problem', 'solve', 'address', 'challenge', 'issue', 'difficult']
    if any(kw in query_lower for kw in problem_keywords):
        return 'problem'
    
    # Methodology queries
    method_keywords = ['method', 'approach', 'how', 'technique', 'algorithm', 'framework']
    if any(kw in query_lower for kw in method_keywords):
        return 'method'
    
    # Results/Findings queries
    results_keywords = ['result', 'finding', 'outcome', 'accuracy', 'performance', 'achieve']
    if any(kw in query_lower for kw in results_keywords):
        return 'results'
    
    # Comparison queries
    compare_keywords = ['compare', 'versus', 'vs', 'better', 'difference', 'similar']
    if any(kw in query_lower for kw in compare_keywords):
        return 'comparison'
    
    return 'generic'


def get_page_boost_for_query(query: str) -> Dict[int, float]:
    """
    Get page boost weights based on query classification.
    This is a fallback when section detection fails.
    """
    query_type = classify_query(query)
    
    if query_type == 'introduction':
        # For introduction queries, boost early pages (1-3)
        return {
            1: 3.0,   # Page 1 highest priority
            2: 2.5,
            3: 2.0,
        }
    elif query_type == 'problem':
        return {1: 2.5, 2: 2.0, 3: 1.5}
    elif query_type == 'method':
        # Method usually in middle
        return {}
    elif query_type == 'results':
        # Results usually in later pages
        return {}
    
    return {}


def get_section_boost_for_query(query: str, available_sections: list = None) -> Dict[str, float]:
    """
    Get section boost weights based on query classification.
    Higher weight = higher priority in retrieval.
    """
    query_type = classify_query(query)
    
    if query_type == 'introduction':
        return {
            "introduction": 2.0,
            "abstract": 2.5,      # Boost abstract heavily for intro queries
            "body": 1.0,
            "methodology": 0.8,
            "results": 0.6,
            "conclusion": 0.8,
            "references": 0.2,
            "acknowledgments": 0.0
        }
    
    elif query_type == 'problem':
        return {
            "introduction": 1.8,  # Problems often stated in intro
            "abstract": 1.5,
            "body": 1.2,
            "methodology": 1.0,
            "results": 0.8,
            "references": 0.3,
            "acknowledgments": 0.0
        }
    
    elif query_type == 'method':
        return {
            "methodology": 2.0,
            "abstract": 1.0,
            "introduction": 1.2,
            "body": 1.3,
            "results": 0.8,
            "references": 0.3,
            "acknowledgments": 0.0
        }
    
    elif query_type == 'results':
        return {
            "results": 2.0,
            "abstract": 1.5,
            "body": 1.0,
            "introduction": 0.8,
            "methodology": 0.8,
            "references": 0.3,
            "acknowledgments": 0.0
        }
    
    elif query_type == 'comparison':
        return {
            "results": 1.8,
            "abstract": 1.5,
            "methodology": 1.2,
            "body": 1.0,
            "introduction": 1.0,
            "references": 0.3,
            "acknowledgments": 0.0
        }
    
    else:  # generic
        return {
            "abstract": 1.5,
            "introduction": 1.3,
            "body": 1.0,
            "methodology": 1.1,
            "results": 1.0,
            "conclusion": 1.0,
            "references": 0.5,
            "acknowledgments": 0.0
        }


def should_include_section(query: str, section: str) -> bool:
    """
    Determine if a section should be included based on query type.
    Can be used to filter out irrelevant sections entirely.
    """
    query_type = classify_query(query)
    
    # Always filter out acknowledgments
    if section == 'acknowledgments':
        return False
    
    # For introduction queries, focus on intro/abstract, but still include body
    if query_type == 'introduction':
        return section in ['introduction', 'abstract', 'body', 'conclusion']
    
    return True
