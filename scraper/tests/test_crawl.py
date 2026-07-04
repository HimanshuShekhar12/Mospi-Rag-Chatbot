"""
scraper/tests/test_crawl.py
----------------------------
Unit tests for scraper/crawl.py — the MoSPI publications API crawler.

These test the pure parsing/transform helpers with canned API items,
plus the Document model and the shared normalisation utilities.
No real HTTP requests are made.
"""

import pytest
from datetime import datetime

from scraper.crawl import (
    build_pdf_url,
    clean_title,
    derive_category,
    parse_publication,
)
from scraper.models import Document
from scraper.utils import normalize_date, normalize_category


# ── Sample API items (shape returned by the real MoSPI API) ───────────────────

ITEM_WITH_PDF = {
    "id": "2851",
    "title": "<p>Estimates of GDP for Q3 2024-25</p>",
    "published_year": "2025-02-28",
    "file_one": {
        "path": "uploads/publications_reports/gdp_q3_2024-25.pdf",
        "type": "document",
        "filemime": "application/pdf",
        "filename": "gdp_q3_2024-25.pdf",
    },
    "file_two": None,
    "redirectional_path": None,
    "is_active": True,
}

ITEM_NO_FILE = {
    "id": "2837",
    "title": "<p>SDG National Indicator Framework 2026</p>",
    "published_year": "2026-06-29",
    "file_one": None,
    "redirectional_path": "/sdg-2026",
    "is_active": True,
}

ITEM_SHORT_TITLE = {
    "id": "1",
    "title": "<p>AB</p>",
    "published_year": "2025-01-01",
    "file_one": {"path": "uploads/x.pdf", "filemime": "application/pdf"},
}


# ── clean_title ───────────────────────────────────────────────────────────────

class TestCleanTitle:
    def test_strips_html_wrapper(self):
        assert clean_title("<p>Estimates of GDP</p>") == "Estimates of GDP"

    def test_decodes_html_entities(self):
        assert clean_title("<p>Data &amp; Analysis</p>") == "Data & Analysis"

    def test_collapses_whitespace(self):
        assert clean_title("<p>A   B\n C</p>") == "A B C"

    def test_empty_input(self):
        assert clean_title(None) == ""
        assert clean_title("") == ""


# ── build_pdf_url ─────────────────────────────────────────────────────────────

class TestBuildPdfUrl:
    def test_relative_path_becomes_absolute(self):
        assert build_pdf_url("uploads/a.pdf") == "https://www.mospi.gov.in/uploads/a.pdf"

    def test_leading_slash_handled(self):
        assert build_pdf_url("/uploads/a.pdf") == "https://www.mospi.gov.in/uploads/a.pdf"

    def test_absolute_url_unchanged(self):
        url = "https://www.mospi.gov.in/uploads/a.pdf"
        assert build_pdf_url(url) == url

    def test_empty_path(self):
        assert build_pdf_url("") == ""


# ── derive_category ───────────────────────────────────────────────────────────

class TestDeriveCategory:
    @pytest.mark.parametrize("title, expected", [
        ("Estimates of GDP for Q3 2024-25", "gdp"),
        ("First Advance Estimate of National Income", "gdp"),
        ("Consumer Price Index December 2024", "cpi"),
        ("Index of Industrial Production November 2024", "iip"),
        ("Periodic Labour Force Survey Annual Report", "plfs"),
        ("Sustainable Development Goals Progress Report", "sdg"),
        ("A Practitioner's Handbook on Data Harmonisation", "handbook"),
        ("Something Completely Unrelated", "publication"),
    ])
    def test_category_from_title(self, title, expected):
        assert derive_category(title) == expected


# ── parse_publication ─────────────────────────────────────────────────────────

class TestParsePublication:
    def test_valid_item_returns_document(self):
        doc = parse_publication(ITEM_WITH_PDF)
        assert doc is not None
        assert doc.title == "Estimates of GDP for Q3 2024-25"
        assert doc.category == "gdp"

    def test_url_is_absolute_pdf(self):
        doc = parse_publication(ITEM_WITH_PDF)
        assert doc.url == "https://www.mospi.gov.in/uploads/publications_reports/gdp_q3_2024-25.pdf"

    def test_date_parsed(self):
        doc = parse_publication(ITEM_WITH_PDF)
        assert doc.date_published is not None
        assert doc.date_published.year == 2025
        assert doc.date_published.month == 2

    def test_file_link_recorded(self):
        doc = parse_publication(ITEM_WITH_PDF)
        assert doc.file_links == [doc.url]

    def test_item_without_pdf_is_skipped(self):
        assert parse_publication(ITEM_NO_FILE) is None

    def test_short_title_is_skipped(self):
        assert parse_publication(ITEM_SHORT_TITLE) is None

    def test_content_hash_is_set(self):
        doc = parse_publication(ITEM_WITH_PDF)
        assert len(doc.content_hash) == 64


# ── Document model ────────────────────────────────────────────────────────────

class TestDocumentModel:
    def test_valid_document_passes(self):
        doc = Document(url="https://mospi.gov.in/test.pdf", title="Test Report 2024")
        assert doc.is_valid() is True

    def test_empty_title_fails_validation(self):
        assert Document(url="https://mospi.gov.in/t.pdf", title="").is_valid() is False

    def test_short_title_fails_validation(self):
        assert Document(url="https://mospi.gov.in/t.pdf", title="AB").is_valid() is False

    def test_empty_url_fails_validation(self):
        assert Document(url="", title="Valid Title Here").is_valid() is False

    def test_hash_computed_on_init(self):
        doc = Document(url="https://mospi.gov.in/t.pdf", title="Test Report")
        assert len(doc.content_hash) == 64

    def test_same_url_title_gives_same_hash(self):
        a = Document(url="https://mospi.gov.in/t.pdf", title="Test")
        b = Document(url="https://mospi.gov.in/t.pdf", title="Test")
        assert a.content_hash == b.content_hash

    def test_different_urls_give_different_hashes(self):
        a = Document(url="https://mospi.gov.in/a.pdf", title="Test")
        b = Document(url="https://mospi.gov.in/b.pdf", title="Test")
        assert a.content_hash != b.content_hash


# ── normalize_date ────────────────────────────────────────────────────────────

class TestNormalizeDate:
    @pytest.mark.parametrize("raw, year, month", [
        ("15 January 2024", 2024, 1),
        ("January 15, 2024", 2024, 1),
        ("2024-01-15", 2024, 1),
        ("15 Jan 2024", 2024, 1),
        ("Jan 2024", 2024, 1),
        ("2025-02-28", 2025, 2),
    ])
    def test_parses_known_formats(self, raw, year, month):
        result = normalize_date(raw)
        assert result is not None
        assert result.year == year
        assert result.month == month

    def test_returns_none_for_garbage(self):
        assert normalize_date("not a date at all") is None

    def test_returns_none_for_empty(self):
        assert normalize_date("") is None
        assert normalize_date(None) is None


# ── normalize_category ────────────────────────────────────────────────────────

class TestNormalizeCategory:
    @pytest.mark.parametrize("raw, expected", [
        ("Press Note", "press-release"),
        ("Report", "report"),
        ("GDP", "gdp"),
        ("CPI", "cpi"),
        ("PLFS", "plfs"),
        (None, "uncategorized"),
        ("", "uncategorized"),
    ])
    def test_category_mapping(self, raw, expected):
        assert normalize_category(raw) == expected
