"""
pipeline/chunker.py
--------------------
Splits long document text into overlapping chunks for RAG indexing.

Assignment requirement:
  "Split long texts into chunks (800-1200 tokens with overlap).
   Keep doc → chunk lineage."

Chunk size is measured in approximate tokens (1 token ≈ 4 characters).
Each chunk carries full lineage: document_id, title, url, category.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from scraper.config import settings
from scraper.models import Document
from scraper.utils import get_logger

logger = get_logger(__name__)

# 1 token ≈ 4 characters (rough but consistent approximation)
CHARS_PER_TOKEN = 4


# ── Chunk model ───────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    A single text chunk ready for embedding.

    Carries full lineage so the RAG system can cite the source document.
    """
    text:        str            # the chunk text
    document_id: Optional[int]  # FK → documents.id
    chunk_index: int            # position within the document (0-based)
    title:       str = ""       # document title (for citations)
    url:         str = ""       # document URL   (for citations)
    category:    str = ""       # document category
    date_published: str = ""    # ISO date string or empty

    @property
    def token_count(self) -> int:
        """Approximate token count for this chunk."""
        return len(self.text) // CHARS_PER_TOKEN

    def __repr__(self) -> str:
        return (
            f"<Chunk doc_id={self.document_id} "
            f"idx={self.chunk_index} "
            f"~{self.token_count} tokens>"
        )


# ── Core splitter ─────────────────────────────────────────────────────────────

def split_text(
    text: str,
    chunk_size:    int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
    min_size:      int = settings.chunk_min_size,
) -> List[str]:
    """
    Split a long text string into overlapping chunks.

    Args:
        text          : raw text to split
        chunk_size    : target chunk size in tokens
        chunk_overlap : overlap between consecutive chunks in tokens
        min_size      : discard chunks smaller than this token count

    Returns:
        List of text strings, each approximately chunk_size tokens.

    Strategy:
        1. Convert token sizes to character sizes
        2. Slide a window across the text with overlap
        3. Try to split at sentence boundaries (". ") to avoid
           cutting mid-sentence when possible
    """
    if not text or not text.strip():
        return []

    chunk_chars   = chunk_size    * CHARS_PER_TOKEN
    overlap_chars = chunk_overlap * CHARS_PER_TOKEN
    min_chars     = min_size      * CHARS_PER_TOKEN
    step          = chunk_chars - overlap_chars

    if step <= 0:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than "
            f"chunk_size ({chunk_size})"
        )

    text   = text.strip()
    chunks = []
    start  = 0

    while start < len(text):
        end = start + chunk_chars

        if end >= len(text):
            # last chunk — take everything remaining
            chunk = text[start:].strip()
        else:
            # try to find a sentence boundary near the end of the window
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + min_chars:
                end   = boundary + 1   # include the period
            chunk = text[start:end].strip()

        if len(chunk) >= min_chars:
            chunks.append(chunk)

        start += step

    return chunks


# ── Document → Chunks ─────────────────────────────────────────────────────────

def chunk_document(
    doc: Document,
    chunk_size:    int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
    min_size:      int = settings.chunk_min_size,
) -> List[Chunk]:
    """
    Convert a Document into a list of Chunk objects.

    Combines title + summary + any available text into one string,
    then splits it into overlapping chunks.

    Each Chunk carries the document's id, title, url, and category
    so the RAG system can cite sources.
    """
    # build the full text: title + summary
    # (PDF full text is stored separately and merged here if present)
    parts = []

    if doc.title:
        parts.append(doc.title.strip())

    if doc.summary:
        parts.append(doc.summary.strip())

    full_text = "\n\n".join(parts)

    if not full_text.strip():
        logger.debug(
            "Skipping document with no text",
            extra={"doc_id": doc.id, "url": doc.url},
        )
        return []

    raw_chunks = split_text(full_text, chunk_size, chunk_overlap, min_size)

    # Guarantee a non-empty document always yields at least one chunk, even if
    # it's shorter than min_size — otherwise short publications would be
    # silently dropped from the index. (split_text itself still filters small
    # trailing fragments within a larger document.)
    if not raw_chunks:
        raw_chunks = [full_text.strip()]

    date_str = (
        doc.date_published.isoformat()
        if doc.date_published
        else ""
    )

    chunks = [
        Chunk(
            text=text,
            document_id=doc.id,
            chunk_index=idx,
            title=doc.title,
            url=doc.url,
            category=doc.category,
            date_published=date_str,
        )
        for idx, text in enumerate(raw_chunks)
    ]

    logger.debug(
        "Document chunked",
        extra={
            "doc_id":   doc.id,
            "title":    doc.title[:40],
            "n_chunks": len(chunks),
        },
    )
    return chunks


def chunk_documents(
    documents: List[Document],
    chunk_size:    int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
    min_size:      int = settings.chunk_min_size,
) -> List[Chunk]:
    """
    Chunk a list of Documents into a flat list of Chunks.

    Used by pipeline/run.py as the main entry point.
    """
    all_chunks: List[Chunk] = []

    for doc in documents:
        chunks = chunk_document(doc, chunk_size, chunk_overlap, min_size)
        all_chunks.extend(chunks)

    logger.info(
        "Chunking complete",
        extra={
            "documents": len(documents),
            "chunks":    len(all_chunks),
        },
    )
    return all_chunks
