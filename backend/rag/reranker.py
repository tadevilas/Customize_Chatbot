"""
Component 4 — Reranker (Cross-Encoder BGE Reranker)

Uses:
  - BAAI/bge-reranker-base (cross-encoder via sentence-transformers)

Purpose:
  The retriever (MMR / BM25 / Hybrid) fetches top-K candidates fast using
  approximate vector search. The reranker then re-scores each candidate by
  reading the query AND the chunk together — much more accurate but slower,
  so it runs only on the small candidate set (e.g. top 10 → reranked → top 3).

  Bi-encoder (retriever):  encodes query and chunk SEPARATELY → fast, approximate
  Cross-encoder (reranker): encodes query + chunk TOGETHER   → slow, accurate

Key operations:
  - rerank()        → reorder a list of Documents by relevance to the query
  - get_reranker()  → load the cross-encoder model (cached after first load)
"""

from typing import List
from langchain_core.documents import Document


RERANKER_MODEL = "BAAI/bge-reranker-base"
_reranker = None   # module-level cache so model loads only once


def get_reranker():
    """
    Load and cache the BGE cross-encoder reranker model.
    Downloads ~1.1 GB on first use, cached locally after that.
    """
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(
    query: str,
    docs: List[Document],
    top_k: int = 5,
) -> List[Document]:
    """
    Rerank a list of Documents by their relevance to the query.

    Steps:
      1. Build (query, chunk_text) pairs for every candidate document
      2. Pass all pairs through the cross-encoder → one relevance score per pair
      3. Sort documents by score descending
      4. Return top_k

    Parameters:
      query  : the user's question
      docs   : candidate documents from retriever (e.g. top-10 from hybrid_search)
      top_k  : how many to return after reranking

    Returns:
      List of Document objects sorted by reranker score (best first),
      with metadata['rerank_score'] added to each.
    """
    if not docs:
        return []

    reranker = get_reranker()

    # Build input pairs: [(query, chunk1_text), (query, chunk2_text), ...]
    pairs = [(query, doc.page_content) for doc in docs]

    # Score all pairs — returns a list of floats (higher = more relevant)
    scores = reranker.predict(pairs)

    # Attach score to metadata and sort
    scored = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    results = []
    for score, doc in scored[:top_k]:
        doc.metadata["rerank_score"] = round(float(score), 4)
        results.append(doc)

    return results
