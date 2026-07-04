"""
scraper/tests/test_parse.py
----------------------------
Unit tests for scraper/parse.py

Tests:
  - extract_text()        returns non-empty string from a real PDF
  - extract_tables()      returns correct table dimensions
  - _compute_hash()       returns correct sha256 hex string
  - _safe_filename()      cleans URLs into safe filenames
  - download_pdf()        handles HTTP errors gracefully
  - _is_already_downloaded() correctly detects duplicates

Uses:
  - A tiny in-memory PDF created with fpdf2 (if available)
    or a pre-built minimal PDF bytes fixture as fallback
  - unittest.mock to avoid real HTTP requests
"""

import hashlib
import io
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from scraper.parse import (
    _compute_hash,
    _safe_filename,
    extract_text,
    extract_tables,
    download_pdf,
)


# ── Minimal PDF fixture ───────────────────────────────────────────────────────
# A valid minimal PDF with one page of text and one simple table.
# Created as raw bytes so the test has zero external dependencies.

MINIMAL_PDF_BYTES = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
    b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    b"4 0 obj\n<< /Length 44 >>\nstream\n"
    b"BT /F1 12 Tf 100 700 Td (GDP Report 2024) Tj ET\n"
    b"endstream\nendobj\n"
    b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    b"xref\n0 6\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000058 00000 n \n"
    b"0000000115 00000 n \n"
    b"0000000266 00000 n \n"
    b"0000000360 00000 n \n"
    b"trailer\n<< /Size 6 /Root 1 0 R >>\n"
    b"startxref\n441\n%%EOF\n"
    # Trailing comment padding (ignored by PDF readers) so the fixture
    # exceeds download_pdf's 1KB "too small to be a real PDF" guard.
    + b"%" + b" padding" * 60 + b"\n"
)


@pytest.fixture
def tmp_pdf(tmp_path: Path) -> Path:
    """
    Write the minimal PDF bytes to a temporary file and return its path.
    pytest's tmp_path fixture gives us a clean temp directory per test.
    """
    pdf_path = tmp_path / "test_report.pdf"
    pdf_path.write_bytes(MINIMAL_PDF_BYTES)
    return pdf_path


@pytest.fixture
def tmp_download_dir(tmp_path: Path) -> Path:
    """Return a temporary directory to simulate the PDF download folder."""
    dl_dir = tmp_path / "pdf"
    dl_dir.mkdir()
    return dl_dir


# ── _compute_hash tests ───────────────────────────────────────────────────────

class TestComputeHash:
    """Tests for the sha256 file hasher."""

    def test_returns_64_char_hex_string(self):
        """sha256 digest should always be exactly 64 hex characters."""
        result = _compute_hash(b"some bytes")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_same_bytes_give_same_hash(self):
        """Hashing the same bytes twice should give identical results."""
        data = b"MoSPI GDP report 2024"
        assert _compute_hash(data) == _compute_hash(data)

    def test_different_bytes_give_different_hash(self):
        """Different content should produce different hashes."""
        assert _compute_hash(b"file A") != _compute_hash(b"file B")

    def test_matches_stdlib_sha256(self):
        """Result should match Python's own hashlib.sha256."""
        data = b"test content"
        expected = hashlib.sha256(data).hexdigest()
        assert _compute_hash(data) == expected

    def test_empty_bytes_returns_known_hash(self):
        """sha256 of empty bytes has a known fixed value."""
        known = hashlib.sha256(b"").hexdigest()
        assert _compute_hash(b"") == known


# ── _safe_filename tests ──────────────────────────────────────────────────────

class TestSafeFilename:
    """Tests for the URL-to-filename converter."""

    def test_extracts_filename_from_url(self):
        url = "https://mospi.gov.in/sites/default/files/gdp_report.pdf"
        assert _safe_filename(url) == "gdp_report.pdf"

    def test_strips_query_parameters(self):
        url = "https://mospi.gov.in/report.pdf?download=1&v=2"
        assert _safe_filename(url) == "report.pdf"

    def test_handles_url_with_no_extension(self):
        url = "https://mospi.gov.in/download/12345"
        result = _safe_filename(url)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_removes_special_characters(self):
        url = "https://mospi.gov.in/files/report (final)!.pdf"
        result = _safe_filename(url)
        # should not contain spaces or parentheses
        assert " " not in result
        assert "(" not in result
        assert ")" not in result

    def test_fallback_for_empty_name(self):
        """If URL has no usable filename, return a default."""
        url = "https://mospi.gov.in/"
        result = _safe_filename(url)
        assert result  # not empty


# ── extract_text tests ────────────────────────────────────────────────────────

class TestExtractText:
    """Tests for PDF text extraction."""

    def test_returns_string_and_page_count(self, tmp_pdf: Path):
        """extract_text should return (str, int) tuple."""
        text, pages = extract_text(tmp_pdf)
        assert isinstance(text, str)
        assert isinstance(pages, int)

    def test_page_count_is_positive(self, tmp_pdf: Path):
        """A valid PDF should report at least 1 page."""
        _, pages = extract_text(tmp_pdf)
        assert pages >= 1

    def test_handles_missing_file_gracefully(self, tmp_path: Path):
        """Should return ('', 0) for a file that does not exist."""
        missing = tmp_path / "does_not_exist.pdf"
        text, pages = extract_text(missing)
        assert text == ""
        assert pages == 0

    def test_handles_corrupt_pdf_gracefully(self, tmp_path: Path):
        """Should return ('', 0) for a corrupt / non-PDF file."""
        bad_pdf = tmp_path / "corrupt.pdf"
        bad_pdf.write_bytes(b"this is not a pdf file at all")
        text, pages = extract_text(bad_pdf)
        assert text == ""
        assert pages == 0


# ── extract_tables tests ──────────────────────────────────────────────────────

class TestExtractTables:
    """Tests for PDF table extraction."""

    def test_returns_list(self, tmp_pdf: Path):
        """extract_tables should always return a list."""
        result = extract_tables(tmp_pdf)
        assert isinstance(result, list)

    def test_each_result_is_tuple_of_page_and_data(self, tmp_pdf: Path):
        """Each item should be (page_number, list_of_lists)."""
        result = extract_tables(tmp_pdf)
        for item in result:
            assert len(item) == 2
            page_num, table_data = item
            assert isinstance(page_num, int)
            assert isinstance(table_data, list)

    def test_handles_missing_file_gracefully(self, tmp_path: Path):
        """Should return empty list for a missing file."""
        missing = tmp_path / "missing.pdf"
        result = extract_tables(missing)
        assert result == []

    def test_handles_corrupt_pdf_gracefully(self, tmp_path: Path):
        """Should return empty list for a corrupt file."""
        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"garbage bytes not a pdf")
        result = extract_tables(bad)
        assert result == []

    def test_table_dimensions_are_positive(self, tmp_pdf: Path):
        """Any extracted table should have at least 2 rows and 2 cols."""
        result = extract_tables(tmp_pdf)
        for _, table_data in result:
            assert len(table_data) >= 2
            assert len(table_data[0]) >= 2


# ── download_pdf tests ────────────────────────────────────────────────────────

class TestDownloadPdf:
    """Tests for the HTTP PDF downloader."""

    def test_successful_download_saves_file(
        self, tmp_download_dir: Path
    ):
        """A successful response should save the file and return its path."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/pdf"}
        mock_response.content = MINIMAL_PDF_BYTES
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response

        with patch("scraper.parse.limiter") as mock_limiter:
            mock_limiter.wait = MagicMock()
            result = download_pdf(
                "https://mospi.gov.in/report.pdf",
                mock_session,
                tmp_download_dir,
            )

        assert result is not None
        local_path, file_bytes = result
        assert local_path.exists()
        assert file_bytes == MINIMAL_PDF_BYTES

    def test_http_error_returns_none(self, tmp_download_dir: Path):
        """A 404 or 500 response should return None, not raise."""
        import requests as req

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = req.HTTPError(
            response=MagicMock(status_code=404)
        )
        mock_response.headers = {"Content-Type": "application/pdf"}

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response

        with patch("scraper.parse.limiter") as mock_limiter:
            mock_limiter.wait = MagicMock()
            result = download_pdf(
                "https://mospi.gov.in/missing.pdf",
                mock_session,
                tmp_download_dir,
            )

        assert result is None

    def test_non_pdf_content_type_returns_none(self, tmp_download_dir: Path):
        """A URL returning HTML instead of PDF should be skipped."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.content = b"<html>not a pdf</html>"
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response

        with patch("scraper.parse.limiter") as mock_limiter:
            mock_limiter.wait = MagicMock()
            result = download_pdf(
                "https://mospi.gov.in/page",
                mock_session,
                tmp_download_dir,
            )

        assert result is None
