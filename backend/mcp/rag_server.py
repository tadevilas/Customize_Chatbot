"""
RAG MCP Tool — wraps hybrid_search + rerank as a callable agent tool.

The LLM (via groq_client) calls this tool when it needs to search
uploaded documents. It returns the top-K reranked chunks formatted
as a string with source citations.
"""

import os
import sys
import asyncio
from typing import Any

# Ensure project root is on path when run as a subprocess
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langchain_core.tools import tool
from backend.config import settings


def _get_chunks_from_chroma(collection_name: str):
    """
    Retrieve all stored documents from ChromaDB to use as BM25 corpus.
    ChromaDB's .get() returns raw dicts — we convert them to Documents.
    """
    import chromadb
    from langchain_core.documents import Document

    client = chromadb.PersistentClient(path=os.path.abspath(settings.CHROMA_DIR))
    try:
        col = client.get_collection(collection_name)
    except Exception:
        return []

    result = col.get(include=["documents", "metadatas"])
    chunks = []
    for text, meta in zip(result["documents"], result["metadatas"]):
        chunks.append(Document(page_content=text, metadata=meta or {}))
    return chunks


@tool
def rag_search(query: str) -> str:
    """
    Search uploaded documents using hybrid retrieval (BM25 + MMR) and
    cross-encoder reranking. Use this tool when the user asks questions
    about documents, reports, manuals, or uploaded files.

    Returns the top relevant passages with source citations.
    """
    from backend.rag.vector_store import hybrid_search, mmr_search
    from backend.rag.reranker import rerank

    collection = settings.RAG_COLLECTION

    # Load all chunks from ChromaDB for BM25
    chunks = _get_chunks_from_chroma(collection)
    if not chunks:
        return "No documents have been uploaded yet. Please upload a document first using the file uploader."

    try:
        # Retrieve candidates: hybrid (BM25 + MMR) for best coverage
        candidates = hybrid_search(
            query,
            chunks,
            k=10,
            collection_name=collection,
            mmr_weight=0.6,
            bm25_weight=0.4,
        )

        # Rerank to get the most accurate top-3
        results = rerank(query, candidates, top_k=3)

    except Exception as e:
        # Fallback to MMR-only if hybrid fails
        try:
            results = mmr_search(query, k=3, fetch_k=10, collection_name=collection)
        except Exception:
            return f"RAG search failed: {str(e)}"

    if not results:
        return "No relevant information found in uploaded documents."

    # Format results with citations
    output_parts = [f"Relevant passages from uploaded documents for query: '{query}'\n"]
    for i, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "?")
        ctype  = doc.metadata.get("content_type", "text")
        score  = doc.metadata.get("rerank_score", "")
        score_str = f" | relevance={score}" if score else ""

        output_parts.append(
            f"[{i}] Source: {source}, Page {page}, Type: {ctype}{score_str}\n"
            f"{doc.page_content}\n"
        )

    return "\n".join(output_parts)
