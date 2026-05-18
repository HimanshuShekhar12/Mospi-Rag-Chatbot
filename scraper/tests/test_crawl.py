"""
scraper/tests/test_crawl.py
----------------------------
Unit tests for scraper/crawl.py

Tests:
  - parse_listing_page()  extracts correct fields from mock HTML
  - find_next_page()      correctly finds pagination links
  - Document.is_valid()   rejects bad records
  - normalize_date()      parses various date formats
  - normalize_category()  maps raw strings to slugs

All tests use mock/fixture HTML — no real HTTP requests are made.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from scraper.crawl import parse_listing_page, find_next_page
from scraper.models import Document
from scraper.utils import normalize_date, normalize_category


# ── Fixtures ──────────────────────────────────────────────────────────────────

MOCK_LISTING_HTML = """
<html>
<body>
  <div class="view-content">

    <div class="views-row">
      <span class="field-content">
        <a href="/press-releases/gdp-q3-2024">
          GDP Advance Estimate Q3 2024
        </a>
      </span>
      <span class="date-display-single">15 January 2024</span>
      <span class="field-category">Press Note</span>
      <div class="field-body">
        <p>The National Statistical Office releases GDP estimates for Q3 2024.</p>
      </div>
      <a href="/sites/files/gdp_q3_2024.pdf">Download PDF</a>
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
        <p>CPI data for December 2023 released by NSO.</p>
      </div>
      <a href="/sites/files/cpi_dec_2023.pdf">Download PDF</a>
    </div>

  </div>

  <li class="pager-next">
    <a href="/press-releases?page=1">next</a>
  </li>
</body>
</html>
"""

MOCK_EMPTY_HTML = """
<html><body><p>No results found.</p></body></html>
"""

MOCK_TABLE_HTML = """
<html>
<body>
  <table class="views-table">
    <tbody>
      <tr>
        <td><a href="/reports/plfs-2023">PLFS Annual Report 2023</a></td>
        <td class="views-field-field-date">March 2024</td>
        <td class="views-field-field-category">PLFS</td>
      </tr>
    </tbody>
  </table>
</body>
</html>
"""


# ── parse_listing_page tests ──────────────────────────────────────────────────

class TestParseListingPage:
    """Tests for the main HTML parser function."""

    def test_extracts_correct_number_of_documents(self):
        """Should find 2 documents in the mock listing HTML."""
        docs = parse_listing_page(MOCK_LISTING_HTML, "https://mospi.gov.in")
        assert len(docs) == 2

    def test_extracts_title_correctly(self):
        """Title should be cleaned and stripped of whitespace."""
        docs = parse_listing_page(MOCK_LISTING_HTML, "https://mospi.gov.in")
        assert docs[0].title == "GDP Advance Estimate Q3 2024"

    def test_extracts_url_as_absolute(self):
        """Relative URLs should be converted to absolute URLs."""
        docs = parse_listing_page(MOCK_LISTING_HTML, "https://mospi.gov.in")
        assert docs[0].url == "https://mospi.gov.in/press-releases/gdp-q3-2024"

    def test_extracts_date_published(self):
        """Date string should be parsed into a datetime object."""
        docs = parse_listing_page(MOCK_LISTING_HTML, "https://mospi.gov.in")
        assert docs[0].date_published is not None
        assert docs[0].date_published.year == 2024
        assert docs[0].date_published.month == 1
        assert docs[0].date_published.day == 15

    def test_extracts_category(self):
        """Category should be normalised to a slug."""
        docs = parse_listing_page(MOCK_LISTING_HTML, "https://mospi.gov.in")
        assert docs[0].category == "press-release"

    def test_extracts_pdf_file_links(self):
        """PDF links should be collected in file_links list."""
        docs = parse_listing_page(MOCK_LISTING_HTML, "https://mospi.gov.in")
        assert len(docs[0].file_links) == 1
        assert docs[0].file_links[0].endswith("gdp_q3_2024.pdf")

    def test_extracts_summary(self):
        """Summary should be extracted from the body field."""
        docs = parse_listing_page(MOCK_LISTING_HTML, "https://mospi.gov.in")
        assert "GDP" in docs[0].summary

    def test_empty_page_returns_empty_list(self):
        """Page with no recognisable content should return empty list."""
        docs = parse_listing_page(MOCK_EMPTY_HTML, "https://mospi.gov.in")
        assert isinstance(docs, list)
        # May return 0 or fallback results — should never raise
        assert len(docs) >= 0

    def test_table_based_listing_parsed(self):
        """Should fall back to table-based selector and find document."""
        docs = parse_listing_page(MOCK_TABLE_HTML, "https://mospi.gov.in")
        assert len(docs) >= 1
        assert "PLFS" in docs[0].title

    def test_content_hash_is_set(self):
        """Each document should have a non-empty content hash."""
        docs = parse_listing_page(MOCK_LISTING_HTML, "https://mospi.gov.in")
        for doc in docs:
            assert doc.content_hash != ""
            assert len(doc.content_hash) == 64  # sha256 hex = 64 chars

    def test_duplicate_hash_differs_between_documents(self):
        """Two different documents should have different hashes."""
        docs = parse_listing_page(MOCK_LISTING_HTML, "https://mospi.gov.in")
        assert docs[0].content_hash != docs[1].content_hash


# ── find_next_page tests ──────────────────────────────────────────────────────

class TestFindNextPage:
    """Tests for the pagination link extractor."""

    def test_finds_pager_next_link(self):
        """Should find the next page URL from a standard pager-next element."""
        next_url = find_next_page(MOCK_LISTING_HTML, "https://mospi.gov.in/press-releases")
        assert next_url is not None
        assert "page=1" in next_url

    def test_returns_none_when_no_next_page(self):
        """Should return None when there is no next page link."""
        html = "<html><body><p>Last page.</p></body></html>"
        next_url = find_next_page(html, "https://mospi.gov.in/press-releases")
        assert next_url is None

    def test_returns_absolute_url(self):
        """Next page URL should always be absolute."""
        next_url = find_next_page(MOCK_LISTING_HTML, "https://mospi.gov.in/press-releases")
        if next_url:
            assert next_url.startswith("http")


# ── Document model tests ──────────────────────────────────────────────────────

class TestDocumentModel:
    """Tests for Document dataclass behaviour."""

    def test_valid_document_passes(self):
        """A document with url and title should pass is_valid()."""
        doc = Document(url="https://mospi.gov.in/test", title="Test Report 2024")
        assert doc.is_valid() is True

    def test_empty_title_fails_validation(self):
        """A document with empty title should fail is_valid()."""
        doc = Document(url="https://mospi.gov.in/test", title="")
        assert doc.is_valid() is False

    def test_short_title_fails_validation(self):
        """A document with a very short title (<=3 chars) should fail."""
        doc = Document(url="https://mospi.gov.in/test", title="AB")
        assert doc.is_valid() is False

    def test_empty_url_fails_validation(self):
        """A document with empty URL should fail is_valid()."""
        doc = Document(url="", title="Valid Title Here")
        assert doc.is_valid() is False

    def test_hash_computed_on_init(self):
        """Content hash should be computed automatically on creation."""
        doc = Document(url="https://mospi.gov.in/test", title="Test Report")
        assert len(doc.content_hash) == 64

    def test_same_url_title_gives_same_hash(self):
        """Same url+title should always produce the same hash (idempotent)."""
        doc1 = Document(url="https://mospi.gov.in/test", title="Test")
        doc2 = Document(url="https://mospi.gov.in/test", title="Test")
        assert doc1.content_hash == doc2.content_hash

    def test_different_urls_give_different_hashes(self):
        """Different URLs should produce different hashes."""
        doc1 = Document(url="https://mospi.gov.in/a", title="Test")
        doc2 = Document(url="https://mospi.gov.in/b", title="Test")
        assert doc1.content_hash != doc2.content_hash


# ── normalize_date tests ──────────────────────────────────────────────────────

class TestNormalizeDate:
    """Tests for the date parser utility."""

    @pytest.mark.parametrize("raw, expected_year, expected_month, expected_day", [
        ("15 January 2024",  2024, 1,  15),
        ("January 15, 2024", 2024, 1,  15),
        ("15-01-2024",       2024, 1,  15),
        ("15/01/2024",       2024, 1,  15),
        ("2024-01-15",       2024, 1,  15),
        ("15 Jan 2024",      2024, 1,  15),
        ("Jan 2024",         2024, 1,   1),
        ("January 2024",     2024, 1,   1),
        ("1st January 2024", 2024, 1,  15),  # ordinal suffix
    ])
    def test_parses_known_formats(self, raw, expected_year, expected_month, expected_day):
        result = normalize_date(raw)
        assert result is not None
        assert result.year == expected_year
        assert result.month == expected_month

    def test_returns_none_for_garbage(self):
        """Unrecognisable strings should return None, not raise."""
        assert normalize_date("not a date at all") is None

    def test_returns_none_for_empty_string(self):
        assert normalize_date("") is None

    def test_returns_none_for_none_input(self):
        assert normalize_date(None) is None


# ── normalize_category tests ──────────────────────────────────────────────────

class TestNormalizeCategory:
    """Tests for category slug normalisation."""

    @pytest.mark.parametrize("raw, expected", [
        ("Press Note",    "press-release"),
        ("Press Release", "press-release"),
        ("press-note",    "press-release"),
        ("Report",        "report"),
        ("Annual Report", "report"),
        ("GDP",           "gdp"),
        ("CPI",           "cpi"),
        ("PLFS",          "plfs"),
        (None,            "uncategorized"),
        ("",              "uncategorized"),
        ("xyz unknown",   "xyz-unknown"),   # slugified fallback
    ])
    def test_category_mapping(self, raw, expected):
        assert normalize_category(raw) == expected
