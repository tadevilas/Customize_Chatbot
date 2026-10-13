"""
Component 1 — Document Loader (Enhanced: Text + Tables + Images)

Strategies per content type:
  Text   → LangChain loaders (PyPDFLoader, Docx2txtLoader, TextLoader)
  Tables → pdfplumber (PDF) / python-docx paragraph scan (DOCX)
           Rendered as markdown table strings so LLM can reason over them
  Images → pytesseract OCR (extracts printed text from embedded images)
           Requires Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki

Returns: list of LangChain Document objects
  doc.page_content  → text / markdown table / OCR text
  doc.metadata      → source, page, type, content_type ("text"|"table"|"image")
"""

import os
import io
from pathlib import Path
from typing import List

from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(text: str, source: str, page: int, file_type: str,
              content_type: str = "text") -> Document:
    return Document(
        page_content=text.strip(),
        metadata={
            "source": source,
            "page": page,
            "type": file_type,
            "content_type": content_type,   # "text" | "table" | "image"
        }
    )


# ---------------------------------------------------------------------------
# PDF loader  (text + tables + images)
# ---------------------------------------------------------------------------

def _load_pdf(file_path: str) -> List[Document]:
    import pdfplumber
    from langchain_community.document_loaders import PyPDFLoader

    source = os.path.basename(file_path)
    docs: List[Document] = []

    # --- Text (via LangChain PyPDFLoader) ---
    lc_loader = PyPDFLoader(file_path)
    for lc_doc in lc_loader.load():
        page_num = lc_doc.metadata.get("page", 0) + 1
        if lc_doc.page_content.strip():
            docs.append(_make_doc(lc_doc.page_content, source, page_num, "pdf", "text"))

    # --- Tables (via pdfplumber) ---
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for table in page.extract_tables():
                if not table:
                    continue
                # Convert to markdown table string
                rows = []
                for i, row in enumerate(table):
                    clean = [str(cell or "").replace("\n", " ").strip() for cell in row]
                    rows.append("| " + " | ".join(clean) + " |")
                    if i == 0:
                        rows.append("|" + "|".join(["---"] * len(row)) + "|")
                md_table = "\n".join(rows)
                if md_table.strip():
                    docs.append(_make_doc(md_table, source, page_num, "pdf", "table"))

    # --- Images / OCR (via pytesseract) ---
    try:
        import pytesseract
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                for img_obj in page.images:
                    try:
                        # Crop the image region from the page
                        x0, top, x1, bottom = (
                            img_obj["x0"], img_obj["top"],
                            img_obj["x1"], img_obj["bottom"]
                        )
                        cropped = page.crop((x0, top, x1, bottom))
                        pil_img = cropped.to_image(resolution=150).original
                        ocr_text = pytesseract.image_to_string(pil_img).strip()
                        if ocr_text and len(ocr_text) > 10:
                            docs.append(_make_doc(
                                f"[Image OCR text]\n{ocr_text}",
                                source, page_num, "pdf", "image"
                            ))
                    except Exception:
                        pass  # skip unreadable images silently
    except Exception:
        pass  # pytesseract not available or no Tesseract binary

    return docs


# ---------------------------------------------------------------------------
# DOCX loader  (text + tables)
# ---------------------------------------------------------------------------

def _load_docx(file_path: str) -> List[Document]:
    """
    Uses python-docx directly for both text and tables.
    Iterates document body in order so text and tables appear in reading sequence.
    """
    import docx as python_docx

    source = os.path.basename(file_path)
    docs: List[Document] = []
    doc = python_docx.Document(file_path)

    # Walk document body elements in order (paragraphs + tables interleaved)
    from docx.text.paragraph import Paragraph
    from docx.table import Table
    para_buffer: List[str] = []

    def flush_paragraphs():
        combined = "\n".join(para_buffer).strip()
        para_buffer.clear()
        if combined:
            docs.append(_make_doc(combined, source, 1, "docx", "text"))

    for element in doc.element.body:
        tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

        if tag == "p":
            para = Paragraph(element, doc)
            text = para.text.strip()
            if text:
                para_buffer.append(text)

        elif tag == "tbl":
            flush_paragraphs()
            table = Table(element, doc)
            rows = []
            for i, row in enumerate(table.rows):
                cells = [cell.text.replace("\n", " ").strip() for cell in row.cells]
                rows.append("| " + " | ".join(cells) + " |")
                if i == 0:
                    rows.append("|" + "|".join(["---"] * len(cells)) + "|")
            md_table = "\n".join(rows)
            if md_table.strip():
                docs.append(_make_doc(md_table, source, 1, "docx", "table"))

    # Flush any remaining paragraphs
    flush_paragraphs()

    return docs


# ---------------------------------------------------------------------------
# TXT loader
# ---------------------------------------------------------------------------

def _load_txt(file_path: str) -> List[Document]:
    from langchain_community.document_loaders import TextLoader
    source = os.path.basename(file_path)
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = source
        doc.metadata.setdefault("page", 1)
        doc.metadata["type"] = "txt"
        doc.metadata["content_type"] = "text"
    return docs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_document(file_path: str) -> List[Document]:
    """
    Auto-detect file type and extract text, tables, and images.

    Returns a list of LangChain Document objects, each tagged with:
      metadata['content_type'] = "text" | "table" | "image"
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return _load_pdf(file_path)
    elif ext == ".docx":
        return _load_docx(file_path)
    elif ext == ".txt":
        return _load_txt(file_path)
    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. Supported: .pdf, .docx, .txt"
        )
