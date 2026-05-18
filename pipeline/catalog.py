"""
pipeline/catalog.py
--------------------
Generates datasets/catalog.json — a manifest of the scraped corpus.

Assignment requirement:
  "Produce a datasets/catalog.json with counts by category/date
   and a manifest of files."

Output structure:
{
  "generated_at": "2024-01-15T10:30:00",
  "total_documents": 42,
  "total_chunks": 187,
  "total_pdfs": 18,
  "by_category": {"press-release": 24, "report": 12, ...},
  "by_year": {"2024": 20, "2023": 15, ...},
  "validation": {"passed": 40, "failed": 2, "pass_rate": "95.2%"},
  "files": [ {manifest of every PDF} ],
  "documents": [ {summary of every document} ]
}
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from scraper.config import settings
from scraper.db import get_all_documents, get_connection
from scraper.models import Document
from scraper.utils import get_logger
from pipeline.validate import ValidationResult, validation_report

logger = get_logger(__name__)

CATALOG_PATH = Path("datasets/catalog.json")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_all_files() -> List[dict]:
    """Fetch all PDF file records from SQLite."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT f.id, f.file_url, f.file_path, f.file_hash,
                   f.file_type, f.pages, f.created_at,
                   d.title as document_title
              FROM files f
              JOIN documents d ON d.id = f.document_id
             ORDER BY f.created_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def _count_by_category(documents: List[Document]) -> dict:
    counts: dict = defaultdict(int)
    for doc in documents:
        counts[doc.category] += 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def _count_by_year(documents: List[Document]) -> dict:
    counts: dict = defaultdict(int)
    for doc in documents:
        if doc.date_published:
            year = str(doc.date_published.year)
        else:
            year = "unknown"
        counts[year] += 1
    return dict(sorted(counts.items(), reverse=True))


def _document_summary(doc: Document) -> dict:
    return {
        "id":             doc.id,
        "title":          doc.title,
        "url":            doc.url,
        "category":       doc.category,
        "date_published": (
            doc.date_published.isoformat() if doc.date_published else None
        ),
        "has_summary":    bool(doc.summary),
        "n_file_links":   len(doc.file_links),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def build_catalog(
    n_chunks: int = 0,
    validation_results: Optional[List[ValidationResult]] = None,
) -> dict:
    """
    Build the catalog dict from the current database state.

    Args:
        n_chunks           : total chunks produced by chunker (pass in from run.py)
        validation_results : results from validate_documents (optional)

    Returns:
        catalog dict (also written to datasets/catalog.json)
    """
    documents = get_all_documents()
    files     = _get_all_files()

    catalog = {
        "generated_at":    datetime.utcnow().isoformat(),
        "total_documents": len(documents),
        "total_chunks":    n_chunks,
        "total_pdfs":      len(files),
        "by_category":     _count_by_category(documents),
        "by_year":         _count_by_year(documents),
        "validation":      (
            validation_report(validation_results)
            if validation_results
            else {}
        ),
        "files":           files,
        "documents":       [_document_summary(d) for d in documents],
    }

    # write to disk
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, default=str)

    logger.info(
        "Catalog written",
        extra={
            "path":      str(CATALOG_PATH),
            "documents": len(documents),
            "chunks":    n_chunks,
        },
    )
    return catalog
