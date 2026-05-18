"""
pipeline/tests/test_validate.py
---------------------------------
Unit tests for pipeline/validate.py

Tests:
  - _check_title()      rejects empty/short titles
  - _check_url()        rejects invalid URLs
  - _check_date()       rejects future dates
  - validate_documents() deduplicates within batch
  - validation_report() produces correct summary stats
"""

import pytest
from datetime import datetime, timedelta

from scraper.models import Document
from pipeline.validate import (
    validate_documents,
    validation_report,
    ValidationResult,
    _check_title,
    _check_url,
    _check_date,
    _check_duplicates,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_document():
    return Document(
        url="https://mospi.gov.in/press-releases/gdp-2024",
        title="GDP First Advance Estimate 2024-25",
        date_published=datetime(2024, 1, 7),
        category="gdp",
        summary="NSO releases GDP estimate for 2024-25.",
    )


@pytest.fixture
def valid_document_2():
    return Document(
        url="https://mospi.gov.in/press-releases/cpi-2024",
        title="Consumer Price Index December 2024",
        date_published=datetime(2024, 1, 12),
        category="cpi",
        summary="CPI data for December 2024.",
    )


# ── _check_title tests ────────────────────────────────────────────────────────

class TestCheckTitle:
    def test_valid_title_passes(self, valid_document):
        result = ValidationResult(document=valid_document)
        _check_title(valid_document, result)
        assert result.passed

    def test_empty_title_fails(self):
        doc = Document(url="https://mospi.gov.in/test", title="")
        result = ValidationResult(document=doc)
        _check_title(doc, result)
        assert not result.passed

    def test_short_title_fails(self):
        doc = Document(url="https://mospi.gov.in/test", title="AB")
        result = ValidationResult(document=doc)
        _check_title(doc, result)
        assert not result.passed

    def test_three_char_title_fails(self):
        doc = Document(url="https://mospi.gov.in/test", title="GDP")
        result = ValidationResult(document=doc)
        _check_title(doc, result)
        assert not result.passed


# ── _check_url tests ──────────────────────────────────────────────────────────

class TestCheckUrl:
    def test_valid_https_url_passes(self, valid_document):
        result = ValidationResult(document=valid_document)
        _check_url(valid_document, result)
        assert result.passed

    def test_empty_url_fails(self):
        doc = Document(url="", title="Valid Title Here")
        result = ValidationResult(document=doc)
        _check_url(doc, result)
        assert not result.passed

    def test_ftp_url_fails(self):
        doc = Document(url="ftp://mospi.gov.in/file", title="Valid Title Here")
        result = ValidationResult(document=doc)
        _check_url(doc, result)
        assert not result.passed

    def test_http_url_passes(self):
        doc = Document(url="http://mospi.gov.in/report", title="Valid Title Here")
        result = ValidationResult(document=doc)
        _check_url(doc, result)
        assert result.passed


# ── _check_date tests ─────────────────────────────────────────────────────────

class TestCheckDate:
    def test_past_date_passes(self, valid_document):
        result = ValidationResult(document=valid_document)
        _check_date(valid_document, result)
        assert result.passed

    def test_future_date_fails(self):
        doc = Document(
            url="https://mospi.gov.in/test",
            title="Future Report Title Here",
            date_published=datetime.utcnow() + timedelta(days=365),
        )
        result = ValidationResult(document=doc)
        _check_date(doc, result)
        assert not result.passed

    def test_none_date_passes(self):
        """Missing date should not fail — just logged as warning."""
        doc = Document(
            url="https://mospi.gov.in/test",
            title="Report Without Date Title",
            date_published=None,
        )
        result = ValidationResult(document=doc)
        _check_date(doc, result)
        assert result.passed


# ── validate_documents tests ──────────────────────────────────────────────────

class TestValidateDocuments:
    def test_valid_documents_all_pass(self, valid_document, valid_document_2):
        valid_docs, results = validate_documents([valid_document, valid_document_2])
        assert len(valid_docs) == 2
        assert all(r.passed for r in results)

    def test_invalid_document_filtered_out(self, valid_document):
        bad_doc = Document(url="", title="X")
        valid_docs, results = validate_documents([valid_document, bad_doc])
        assert len(valid_docs) == 1
        assert valid_docs[0].url == valid_document.url

    def test_empty_list_returns_empty(self):
        valid_docs, results = validate_documents([])
        assert valid_docs == []
        assert results == []

    def test_duplicate_urls_flagged(self, valid_document):
        """Same URL twice should flag the second as duplicate."""
        doc2 = Document(
            url=valid_document.url,   # same URL
            title="Different Title For Testing Dedup",
        )
        valid_docs, results = validate_documents([valid_document, doc2])
        failed = [r for r in results if not r.passed]
        assert len(failed) >= 1

    def test_results_count_matches_input(self, valid_document, valid_document_2):
        """Result list should have same length as input list."""
        _, results = validate_documents([valid_document, valid_document_2])
        assert len(results) == 2


# ── validation_report tests ───────────────────────────────────────────────────

class TestValidationReport:
    def test_report_has_required_keys(self, valid_document, valid_document_2):
        _, results = validate_documents([valid_document, valid_document_2])
        report = validation_report(results)
        assert "total"    in report
        assert "passed"   in report
        assert "failed"   in report
        assert "pass_rate" in report

    def test_pass_rate_format(self, valid_document):
        _, results = validate_documents([valid_document])
        report = validation_report(results)
        assert "%" in report["pass_rate"]

    def test_totals_are_correct(self, valid_document, valid_document_2):
        _, results = validate_documents([valid_document, valid_document_2])
        report = validation_report(results)
        assert report["total"]  == 2
        assert report["passed"] == 2
        assert report["failed"] == 0
