"""
Component 3 — Vector Store (LangChain + ChromaDB + BM25 + MMR Hybrid Search)

Uses:
  - HuggingFaceEmbeddings (BAAI/bge-small-en-v1.5) for semantic search
  - Chroma (via langchain-chroma) as the local persistent vector database
  - BM25Retriever (rank-bm25) for keyword-based search
  - MMR (Maximal Marginal Relevance) for diverse, non-redundant retrieval
  - Hybrid search: BM25 + MMR merged with Reciprocal Rank Fusion

Key operations:
  - add_documents()       → embed chunks and store in ChromaDB
  - mmr_search()          → MMR search (relevant + diverse, no duplicates)
  - search_with_scores()  → plain similarity search + score (for comparison)
  - bm25_search()         → keyword search (BM25)
  - hybrid_search()       → BM25 + MMR merged via Reciprocal Rank Fusion
  - get_store()           → load or create a ChromaDB collection
  - clear()               → delete all documents in a collection
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

# Default paths / settings — use config so it matches production
from backend.config import settings as _settings
CHROMA_DIR = os.path.abspath(_settings.CHROMA_DIR)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the BGE embedding model.
    Downloads on first use (~130 MB), cached locally after that.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity works best normalised
    )


def get_store(collection_name: str = "rag_documents") -> Chroma:
    """
    Load (or create) a persistent ChromaDB collection.
    Data is saved to chroma_db/ in the project root.
    """
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=os.path.abspath(CHROMA_DIR),
    )


def add_documents(chunks: List[Document], collection_name: str = "rag_documents") -> Chroma:
    """
    Embed a list of LangChain Document chunks and store them in ChromaDB.
    Returns the Chroma store instance.
    """
    store = get_store(collection_name)
    store.add_documents(chunks)
    return store


def mmr_search(
    query: str,
    k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    collection_name: str = "rag_documents",
) -> List[Document]:
    """
    MMR (Maximal Marginal Relevance) search.

    Unlike plain similarity search which can return near-duplicate chunks,
    MMR balances two goals:
      - Relevance   : how similar the chunk is to the query
      - Diversity   : how different the chunk is from already-selected chunks

    Parameters:
      k           : number of chunks to return
      fetch_k     : how many candidates to fetch from ChromaDB before MMR reranking
                    (should be > k; more candidates → better diversity)
      lambda_mult : diversity tuning (0.0 = max diversity, 1.0 = max relevance)
                    0.5 is a balanced default

    Algorithm:
      1. Fetch `fetch_k` candidates from ChromaDB by similarity
      2. Iteratively pick the next chunk that maximises:
             lambda * similarity(chunk, query)
           - (1-lambda) * max_similarity(chunk, already_selected)
    """
    store = get_store(collection_name)
    return store.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )


def search_with_scores(query: str, k: int = 5, collection_name: str = "rag_documents") -> List[tuple]:
    """
    Plain similarity search with scores — useful for comparison against MMR.
    Returns: [(Document, score), ...] — lower L2 distance = more similar.
    """
    store = get_store(collection_name)
    return store.similarity_search_with_score(query, k=k)


def bm25_search(query: str, chunks: List[Document], k: int = 5) -> List[Document]:
    """
    Keyword-based BM25 search over a list of Document chunks.

    BM25 scores documents by exact term frequency — great for:
      - Specific product names, codes, numbers
      - Exact phrase matching
      - Cases where semantic search may miss precise keywords

    Note: BM25 is stateless — it re-indexes `chunks` on every call.
    For production use, cache the retriever outside this function.
    """
    retriever = BM25Retriever.from_documents(chunks, k=k)
    return retriever.invoke(query)


def hybrid_search(
    query: str,
    chunks: List[Document],
    k: int = 5,
    collection_name: str = "rag_documents",
    mmr_weight: float = 0.6,
    bm25_weight: float = 0.4,
    lambda_mult: float = 0.5,
) -> List[Document]:
    """
    Hybrid search: BM25 (keyword) + MMR (semantic + diverse).

    Uses Reciprocal Rank Fusion (RRF) to merge both result lists:
      score(doc) = mmr_weight  * 1/(mmr_rank  + 1)
                 + bm25_weight * 1/(bm25_rank + 1)

    Why MMR instead of plain similarity?
      Plain similarity can return 3 near-identical chunks all saying the same thing.
      MMR ensures each selected chunk adds new information.

    mmr_weight + bm25_weight should sum to 1.0
    """
    fetch_k = k * 2  # fetch extra candidates before fusion

    # --- MMR results (diverse semantic) ---
    mmr_results = mmr_search(query, k=fetch_k, fetch_k=fetch_k * 2,
                             lambda_mult=lambda_mult, collection_name=collection_name)

    # --- BM25 results (keyword) ---
    bm25_results = bm25_search(query, chunks, k=fetch_k)

    # --- Reciprocal Rank Fusion scoring ---
    scores: dict = {}
    doc_map: dict = {}

    for rank, doc in enumerate(mmr_results):
        key = doc.page_content.strip()
        scores[key] = scores.get(key, 0) + mmr_weight * (1.0 / (rank + 1))
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_results):
        key = doc.page_content.strip()
        scores[key] = scores.get(key, 0) + bm25_weight * (1.0 / (rank + 1))
        doc_map[key] = doc

    # Sort by combined score descending, return top-K
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)[:k]
    return [doc_map[key] for key in sorted_keys]


def clear(collection_name: str = "rag_documents") -> None:
    """
    Delete all documents in a collection by deleting and recreating it.
    Safe to call even if the collection does not exist yet.
    """
    import chromadb
    client = chromadb.PersistentClient(path=os.path.abspath(CHROMA_DIR))
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)
    # Recreate as empty so the collection is ready for fresh inserts
    client.get_or_create_collection(collection_name)
    print(f"Collection '{collection_name}' cleared and ready.")
