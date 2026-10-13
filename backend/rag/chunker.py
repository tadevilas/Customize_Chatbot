"""
Component 2 — Chunker (LangChain-based)
Uses LangChain RecursiveCharacterTextSplitter.

Takes load_document() output (list of LangChain Documents)
and splits each into smaller overlapping chunks.

Config: chunk_size=1000, chunk_overlap=200
Returns: list of LangChain Document objects (each is one chunk)
  doc.page_content          → chunk text
  doc.metadata.source       → original filename
  doc.metadata.page         → page number it came from
  doc.metadata.chunk_index  → chunk number within that page
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Split a list of LangChain Documents into smaller chunks.

    Uses RecursiveCharacterTextSplitter which tries to split on:
      paragraphs → sentences → words → characters
    so chunks break at natural language boundaries where possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,   # adds 'start_index' to metadata for debugging
    )

    chunks = splitter.split_documents(docs)

    # Add chunk_index per source+page group for traceability
    counter: dict = {}
    for chunk in chunks:
        key = (chunk.metadata.get("source"), chunk.metadata.get("page"))
        counter[key] = counter.get(key, 0)
        chunk.metadata["chunk_index"] = counter[key]
        counter[key] += 1

    return chunks
