"""
scraper/tests/test_integration.py
-----------------------------------
Integration test — the full crawl pipeline end to end.

What this covers:
  1. The MoSPI listing API is mocked (no real HTTP).
  2. PDF download + text extraction are mocked with canned bytes/text.
  3. crawl() runs and writes to a real (temporary) SQLite database.
  4. We assert the DB holds the correct records, and dedup works.

No internet connection is required.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scraper.crawl import crawl
from scraper.db import init_db, get_all_documents, get_run_summary


# ── One page of the real API response shape ───────────────────────────────────

API_PAGE = {
    "status": True,
    "data": [
        {
            "id": "1",
            "title": "<p>Estimates of GDP for Q3 2024-25</p>",
            "published_year": "2025-02-28",
            "file_one": {"path": "uploads/gdp_q3.pdf", "filemime": "application/pdf"},
        },
        {
            "id": "2",
            "title": "<p>Consumer Price Index December 2024</p>",
            "published_year": "2025-01-13",
            "file_one": {"path": "uploads/cpi_dec.pdf", "filemime": "application/pdf"},
        },
        {
            # metadata-only entry (no PDF) — should be skipped
            "id": "3",
            "title": "<p>SDG Progress Report 2026</p>",
            "published_year": "2026-06-29",
            "file_one": None,
            "redirectional_path": "/sdg-2026",
        },
    ],
    "pagination": {"currentPage": 1, "totalPages": 1, "pageSize": 3},
}

SAMPLE_TEXT = "This is a MoSPI statistical publication. " * 40  # > 200 chars


@pytest.fixture(autouse=True)
def use_temp_db(tmp_path: Path, monkeypatch):
    """Redirect all DB + file writes to temp dirs so real data is untouched."""
    temp_db = str(tmp_path / "test_mospi.db")
    # db_path is a read-only property that returns database_url, so patch the field.
    monkeypatch.setattr("scraper.config.settings.database_url", temp_db)
    monkeypatch.setattr("scraper.crawl.settings.pdf_download_dir", str(tmp_path / "pdf"))
    init_db()
    yield temp_db


def _mock_api_session(page_body=API_PAGE):
    """A requests.Session whose .post() returns the given API page."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=page_body)

    session = MagicMock()
    session.post = MagicMock(return_value=resp)
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    return session


def _run_crawl():
    """Run crawl() with the API mocked and the PDF layer stubbed out."""
    with patch("scraper.crawl.requests.Session", return_value=_mock_api_session()), \
         patch("scraper.crawl.is_allowed", return_value=True), \
         patch("scraper.crawl.limiter") as mock_limiter, \
         patch("scraper.crawl.download_pdf", return_value=b"%PDF-1.4 fake bytes"), \
         patch("scraper.crawl.extract_pdf_text", return_value=(SAMPLE_TEXT, 5)), \
         patch("scraper.crawl._save_pdf_bytes", return_value=Path("fake.pdf")):
        mock_limiter.wait = MagicMock()
        return crawl(max_pages=1, page_size=3)


# ── Integration tests ─────────────────────────────────────────────────────────

class TestFullCrawlPipeline:

    def test_documents_saved_to_database(self):
        _run_crawl()
        docs = get_all_documents()
        assert len(docs) == 2  # 2 PDFs ingested, 1 metadata-only skipped

    def test_summary_counts_are_correct(self):
        summary = _run_crawl()
        assert summary["total_fetched"] == 2   # only the 2 PDF-backed items
        assert summary["total_saved"] == 2
        assert summary["total_failed"] == 0
        assert summary["pages_crawled"] == 1

    def test_document_fields_are_correct(self):
        _run_crawl()
        docs = get_all_documents()
        titles = [d.title for d in docs]
        assert "Estimates of GDP for Q3 2024-25" in titles
        assert "Consumer Price Index December 2024" in titles

    def test_extracted_text_stored_as_summary(self):
        _run_crawl()
        docs = get_all_documents()
        for doc in docs:
            assert len(doc.summary) > 200
            assert "MoSPI" in doc.summary

    def test_categories_derived(self):
        _run_crawl()
        docs = get_all_documents()
        cats = {d.title: d.category for d in docs}
        assert cats["Estimates of GDP for Q3 2024-25"] == "gdp"
        assert cats["Consumer Price Index December 2024"] == "cpi"

    def test_deduplication_prevents_duplicates(self):
        _run_crawl()
        summary = _run_crawl()  # second run on identical data
        docs = get_all_documents()
        assert len(docs) == 2           # still 2, not 4
        assert summary["total_skipped"] == 2

    def test_files_table_populated(self):
        _run_crawl()
        stats = get_run_summary()
        assert stats["total_documents"] == 2
        assert stats["total_files"] == 2

    def test_robots_blocked_returns_nothing(self):
        with patch("scraper.crawl.requests.Session", return_value=_mock_api_session()), \
             patch("scraper.crawl.is_allowed", return_value=False):
            crawl(max_pages=1, page_size=3)
        assert len(get_all_documents()) == 0
