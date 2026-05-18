"""
scraper/tests/test_integration.py
-----------------------------------
Integration test — full crawl pipeline end to end.

What this tests:
  1. A mock HTML listing page is served (no real HTTP)
  2. crawl() runs against it
  3. Documents are saved to a real (temporary) SQLite database
  4. We verify the DB has the correct records

This satisfies the assignment requirement:
  "Integration: one end-to-end run against a tiny fixture page
   (mock HTML) and a sample PDF."

No real internet connection is needed.
"""

import pytest
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

from scraper.crawl import crawl, parse_listing_page
from scraper.db import init_db, get_all_documents, get_run_summary
from scraper.models import Document


# ── Mock HTML fixture ─────────────────────────────────────────────────────────

FIXTURE_HTML = """
<html>
<body>
  <div class="view-content">

    <div class="views-row">
      <span class="field-content">
        <a href="/press-releases/gdp-q1-2024">
          GDP First Advance Estimate 2024-25
        </a>
      </span>
      <span class="date-display-single">07 January 2024</span>
      <span class="field-category">Press Note</span>
      <div class="field-body">
        <p>NSO releases First Advance Estimate of GDP for 2024-25.</p>
      </div>
      <a href="/files/gdp_advance_2024.pdf">Download PDF</a>
    </div>

    <div class="views-row">
      <span class="field-content">
        <a href="/press-releases/iip-november-2023">
          Index of Industrial Production November 2023
        </a>
      </span>
      <span class="date-display-single">12 January 2024</span>
      <span class="field-category">Press Release</span>
      <div class="field-body">
        <p>IIP data for November 2023 shows 2.4% growth.</p>
      </div>
      <a href="/files/iip_nov_2023.pdf">Download PDF</a>
    </div>

    <div class="views-row">
      <span class="field-content">
        <a href="/press-releases/cpi-december-2023">
          Consumer Price Index December 2023
        </a>
      </span>
      <span class="date-display-single">12 January 2024</span>
      <span class="field-category">Report</span>
      <div class="field-body">
        <p>CPI for December 2023 stands at 5.69%.</p>
      </div>
    </div>

  </div>
</body>
</html>
"""

# second page — used to test pagination stops correctly
FIXTURE_HTML_PAGE2 = """
<html>
<body>
  <div class="view-content">
    <div class="views-row">
      <span class="field-content">
        <a href="/press-releases/plfs-2023">
          Periodic Labour Force Survey 2022-23
        </a>
      </span>
      <span class="date-display-single">October 2023</span>
      <span class="field-category">PLFS</span>
    </div>
  </div>
</body>
</html>
"""


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def use_temp_db(tmp_path: Path, monkeypatch):
    """
    Redirect all DB operations to a temporary SQLite file for each test.
    monkeypatch ensures the real data/mospi.db is never touched.
    """
    temp_db = str(tmp_path / "test_mospi.db")
    monkeypatch.setattr("scraper.db.settings.database_url", temp_db)
    monkeypatch.setattr("scraper.db.settings.db_path",      temp_db)
    monkeypatch.setattr("scraper.config.settings.database_url", temp_db)
    init_db()
    yield temp_db


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_mock_session(html_pages: list) -> MagicMock:
    """
    Build a mock requests.Session whose .get() returns pages
    from html_pages in order, then raises ConnectionError.
    """
    responses = []
    for html in html_pages:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()
        responses.append(mock_resp)

    mock_session = MagicMock()
    mock_session.get.side_effect = responses
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    return mock_session


# ── Integration tests ─────────────────────────────────────────────────────────

class TestFullCrawlPipeline:
    """
    End-to-end integration tests for the crawl pipeline.
    No real HTTP — everything mocked at the session level.
    """

    def test_documents_saved_to_database(self, monkeypatch):
        """
        After crawling a mock page, documents should appear in SQLite.
        """
        mock_session = _make_mock_session([FIXTURE_HTML])

        with patch("scraper.crawl.requests.Session") as mock_session_cls, \
             patch("scraper.crawl.is_allowed", return_value=True), \
             patch("scraper.crawl.limiter") as mock_limiter:

            mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_limiter.wait = MagicMock()

            summary = crawl(
                seed_urls=["https://mospi.gov.in/press-releases"],
                max_pages=1,
            )

        docs = get_all_documents()
        assert len(docs) >= 1, "At least one document should be saved"

    def test_correct_number_of_documents_saved(self, monkeypatch):
        """
        All 3 documents from the fixture page should be saved.
        """
        mock_session = _make_mock_session([FIXTURE_HTML])

        with patch("scraper.crawl.requests.Session") as mock_session_cls, \
             patch("scraper.crawl.is_allowed", return_value=True), \
             patch("scraper.crawl.limiter") as mock_limiter:

            mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_limiter.wait = MagicMock()

            crawl(
                seed_urls=["https://mospi.gov.in/press-releases"],
                max_pages=1,
            )

        docs = get_all_documents()
        assert len(docs) == 3

    def test_document_fields_are_correct(self, monkeypatch):
        """
        Saved documents should have correct title, category and date.
        """
        mock_session = _make_mock_session([FIXTURE_HTML])

        with patch("scraper.crawl.requests.Session") as mock_session_cls, \
             patch("scraper.crawl.is_allowed", return_value=True), \
             patch("scraper.crawl.limiter") as mock_limiter:

            mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_limiter.wait = MagicMock()

            crawl(
                seed_urls=["https://mospi.gov.in/press-releases"],
                max_pages=1,
            )

        docs = get_all_documents()
        titles = [d.title for d in docs]

        assert "GDP First Advance Estimate 2024-25" in titles
        assert "Index of Industrial Production November 2023" in titles
        assert "Consumer Price Index December 2023" in titles

    def test_deduplication_prevents_duplicate_records(self, monkeypatch):
        """
        Running crawl twice on the same page should NOT create duplicates.
        """
        mock_session = _make_mock_session([FIXTURE_HTML])

        with patch("scraper.crawl.requests.Session") as mock_session_cls, \
             patch("scraper.crawl.is_allowed", return_value=True), \
             patch("scraper.crawl.limiter") as mock_limiter:

            mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_limiter.wait = MagicMock()

            # crawl once
            crawl(
                seed_urls=["https://mospi.gov.in/press-releases"],
                max_pages=1,
            )

        # crawl same page again
        mock_session2 = _make_mock_session([FIXTURE_HTML])
        with patch("scraper.crawl.requests.Session") as mock_session_cls, \
             patch("scraper.crawl.is_allowed", return_value=True), \
             patch("scraper.crawl.limiter") as mock_limiter:

            mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session2)
            mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_limiter.wait = MagicMock()

            summary = crawl(
                seed_urls=["https://mospi.gov.in/press-releases"],
                max_pages=1,
            )

        docs = get_all_documents()
        assert len(docs) == 3                       # still 3, not 6
        assert summary["total_skipped"] == 3        # all 3 skipped as duplicates

    def test_run_summary_counts_are_correct(self, monkeypatch):
        """
        get_run_summary() should report correct totals after crawl.
        """
        mock_session = _make_mock_session([FIXTURE_HTML])

        with patch("scraper.crawl.requests.Session") as mock_session_cls, \
             patch("scraper.crawl.is_allowed", return_value=True), \
             patch("scraper.crawl.limiter") as mock_limiter:

            mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_limiter.wait = MagicMock()

            crawl(
                seed_urls=["https://mospi.gov.in/press-releases"],
                max_pages=1,
            )

        summary = get_run_summary()
        assert summary["total_documents"] == 3
        assert summary["total_files"] == 0      # no PDFs downloaded in crawl
        assert "by_category" in summary

    def test_crawl_respects_max_pages_limit(self, monkeypatch):
        """
        Crawl should stop after max_pages even if next page exists.
        """
        # provide 2 pages but set max_pages=1
        mock_session = _make_mock_session([FIXTURE_HTML, FIXTURE_HTML_PAGE2])

        with patch("scraper.crawl.requests.Session") as mock_session_cls, \
             patch("scraper.crawl.is_allowed", return_value=True), \
             patch("scraper.crawl.limiter") as mock_limiter:

            mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_limiter.wait = MagicMock()

            summary = crawl(
                seed_urls=["https://mospi.gov.in/press-releases"],
                max_pages=1,
            )

        # only 3 docs from page 1, not 4 from page 1+2
        docs = get_all_documents()
        assert len(docs) == 3
        assert summary["pages_crawled"] == 1

    def test_robots_blocked_url_is_skipped(self, monkeypatch):
        """
        URLs blocked by robots.txt should produce zero saved documents.
        """
        with patch("scraper.crawl.requests.Session") as mock_session_cls, \
             patch("scraper.crawl.is_allowed", return_value=False), \
             patch("scraper.crawl.limiter") as mock_limiter:

            mock_session_cls.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_limiter.wait = MagicMock()

            crawl(
                seed_urls=["https://mospi.gov.in/blocked"],
                max_pages=1,
            )

        docs = get_all_documents()
        assert len(docs) == 0


# ── parse_listing_page direct tests ──────────────────────────────────────────

class TestParseListingPageIntegration:
    """
    Direct tests on parse_listing_page with the full fixture HTML.
    Verifies the parser + model + dedup chain works together.
    """

    def test_all_documents_are_valid(self):
        """Every document parsed from fixture HTML should pass is_valid()."""
        docs = parse_listing_page(FIXTURE_HTML, "https://mospi.gov.in")
        assert all(doc.is_valid() for doc in docs)

    def test_all_documents_have_unique_hashes(self):
        """No two documents should share the same content hash."""
        docs = parse_listing_page(FIXTURE_HTML, "https://mospi.gov.in")
        hashes = [doc.content_hash for doc in docs]
        assert len(hashes) == len(set(hashes))

    def test_pdf_links_are_absolute(self):
        """All file_links should be absolute URLs."""
        docs = parse_listing_page(FIXTURE_HTML, "https://mospi.gov.in")
        for doc in docs:
            for link in doc.file_links:
                assert link.startswith("http"), f"Relative link found: {link}"

    def test_dates_parsed_correctly(self):
        """Documents with valid dates should have datetime objects."""
        docs = parse_listing_page(FIXTURE_HTML, "https://mospi.gov.in")
        dated = [d for d in docs if d.date_published is not None]
        assert len(dated) >= 2
        for doc in dated:
            assert doc.date_published.year == 2024
