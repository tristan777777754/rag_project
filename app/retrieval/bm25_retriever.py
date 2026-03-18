"""
BM25 Retriever for keyword-based document retrieval.
Complements embedding retrieval with exact keyword matching.
"""
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
import re


class BM25Retriever:
    """
    BM25-based retriever for keyword matching.
    Works alongside embedding retrieval for hybrid search.
    """
    
    def __init__(self, chunks: List[Dict[str, Any]]):
        """
        Initialize BM25 retriever with document chunks.
        
        Args:
            chunks: List of chunk dicts with 'text', 'page', etc.
        """
        self.chunks = chunks
        self.chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        # Tokenize all chunks
        self.tokenized_chunks = [
            self._tokenize(chunk["text"]) 
            for chunk in chunks
        ]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_chunks)
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for BM25.
        - Lowercase
        - Remove punctuation
        - Split on whitespace
        """
        # Lowercase and remove non-alphanumeric (keep spaces)
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        # Split and filter empty
        tokens = [t for t in text.split() if t.strip()]
        return tokens
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using BM25.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
        
        Returns:
            List of chunks with BM25 scores
        """
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        if not tokenized_query:
            return []
        
        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)), 
            key=lambda i: scores[i], 
            reverse=True
        )[:top_k]
        
        # Return chunks with scores
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return positive scores
                chunk = self.chunks[idx].copy()
                chunk["bm25_score"] = float(scores[idx])
                chunk["retrieval_method"] = "bm25"
                results.append(chunk)
        
        return results
    
    def get_chunk_by_id(self, chunk_id: str) -> Dict[str, Any]:
        """Get a chunk by its ID."""
        if chunk_id in self.chunk_ids:
            idx = self.chunk_ids.index(chunk_id)
            return self.chunks[idx]
        return None


def create_bm25_index(chunks: List[Dict[str, Any]]) -> BM25Retriever:
    """
    Convenience function to create a BM25 retriever from chunks.
    """
    return BM25Retriever(chunks)
