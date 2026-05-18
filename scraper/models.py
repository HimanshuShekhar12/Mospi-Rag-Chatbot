"""
scraper/models.py
-----------------
Dataclass models representing the core entities scraped from MoSPI.

Three models:
  - Document   : a report or press note (the main entity)
  - PDFFile    : a PDF file linked to a document
  - ExtractedTable : a table pulled out of a PDF page
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


# ── Document ──────────────────────────────────────────────────────────────────

@dataclass
class Document:
    """
    Represents one publication/press-note scraped from a MoSPI listing page.

    Fields map directly to the assignment's required metadata model:
      id, title, date_published, url, category, summary, file_links[]
    """

    url: str                                   # canonical page URL
    title: str                                 # cleaned publication title
    date_published: Optional[datetime] = None  # parsed from page (may be None)
    category: str = "uncategorized"            # subject / section tag
    summary: str = ""                          # abstract or first paragraph
    file_links: List[str] = field(default_factory=list)  # PDF / Excel URLs

    # filled automatically — do not set manually
    id: Optional[int] = None                   # SQLite row id (set after insert)
    content_hash: str = ""                     # sha256(url + title) for dedup
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        """Compute content fingerprint right after construction."""
        self.title = self.title.strip()
        self.summary = self.summary.strip()
        if not self.content_hash:
            raw = f"{self.url}|{self.title}"
            self.content_hash = hashlib.sha256(raw.encode()).hexdigest()

    def is_valid(self) -> bool:
        """
        Minimal sanity check used by the pipeline validator.
        Returns True only if the record has the bare minimum data.
        """
        return bool(self.url and self.title and len(self.title) > 3)

    def __repr__(self) -> str:
        date_str = (
            self.date_published.strftime("%Y-%m-%d")
            if self.date_published
            else "unknown date"
        )
        return f"<Document [{date_str}] {self.title[:60]}>"


# ── PDFFile ───────────────────────────────────────────────────────────────────

@dataclass
class PDFFile:
    """
    Represents a single PDF file downloaded from a Document's file_links.

    Stores both the remote URL and the local path where it was saved,
    plus a sha256 hash of the file bytes for deduplication.
    """

    document_id: int        # foreign key → Document.id
    file_url: str           # original remote URL
    file_path: str          # local path under data/raw/pdf/
    file_type: str = "pdf"  # always pdf for now; could be xlsx in future

    # filled after download + extraction
    id: Optional[int] = None
    file_hash: str = ""     # sha256 of raw file bytes
    pages: int = 0          # total pages in PDF
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<PDFFile pages={self.pages} {self.file_path}>"


# ── ExtractedTable ────────────────────────────────────────────────────────────

@dataclass
class ExtractedTable:
    """
    Represents one table extracted from a PDF page via pdfplumber.

    table_data holds the raw list-of-lists (rows × columns).
    Serialised to JSON before storing in SQLite.
    """

    document_id: int          # foreign key → Document.id
    source_file_id: int       # foreign key → PDFFile.id
    page_number: int          # which PDF page the table came from
    table_data: List[List]    # raw rows — list of lists

    # filled automatically
    id: Optional[int] = None
    n_rows: int = 0
    n_cols: int = 0

    def __post_init__(self) -> None:
        """Compute dimensions from table_data."""
        if self.table_data:
            self.n_rows = len(self.table_data)
            self.n_cols = max((len(row) for row in self.table_data), default=0)

    def __repr__(self) -> str:
        return (
            f"<ExtractedTable page={self.page_number} "
            f"{self.n_rows}×{self.n_cols}>"
        )
