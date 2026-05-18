"""
pipeline/validate.py
---------------------
Data quality validation for scraped MoSPI documents.

Checks performed on every document:
  - Non-empty title (length > 3)
  - Valid publication date (not in the future)
  - URL format is valid
  - No duplicate URLs in the batch
  - Summary is not suspiciously short

Returns a ValidationResult per document with pass/fail + reasons.
Used by pipeline/run.py before chunking begins.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple
from urllib.parse import urlparse

from scraper.models import Document
from scraper.utils import get_logger

logger = get_logger(__name__)


# ── Result model ──────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """
    Holds the outcome of validating one Document.

    Attributes:
        document  : the Document that was checked
        passed    : True if all checks passed
        failures  : list of human-readable failure reasons
    """
    document: Document
    passed: bool = True
    failures: List[str] = field(default_factory=list)

    def fail(self, reason: str) -> None:
        """Mark this result as failed and record the reason."""
        self.passed = False
        self.failures.append(reason)

    def __repr__(self) -> str:
        status = "PASS" if self.passed else f"FAIL({'; '.join(self.failures)})"
        return f"<ValidationResult {status} — {self.document.title[:40]}>"


# ── Individual checks ─────────────────────────────────────────────────────────

def _check_title(doc: Document, result: ValidationResult) -> None:
    """Title must be non-empty and longer than 3 characters."""
    if not doc.title or len(doc.title.strip()) <= 3:
        result.fail(f"Title too short or empty: '{doc.title}'")


def _check_url(doc: Document, result: ValidationResult) -> None:
    """URL must be a valid http/https URL."""
    if not doc.url:
        result.fail("URL is empty")
        return
    try:
        parsed = urlparse(doc.url)
        if parsed.scheme not in ("http", "https"):
            result.fail(f"Invalid URL scheme: '{parsed.scheme}'")
        if not parsed.netloc:
            result.fail(f"URL has no domain: '{doc.url}'")
    except Exception:
        result.fail(f"Malformed URL: '{doc.url}'")


def _check_date(doc: Document, result: ValidationResult) -> None:
    """
    Date must not be in the future.
    Missing date is a warning — not a hard failure (many MoSPI pages omit dates).
    """
    if doc.date_published is None:
        # soft warning — logged but does not fail validation
        logger.debug("Missing date", extra={"url": doc.url})
        return
    if doc.date_published > datetime.utcnow():
        result.fail(
            f"Publication date is in the future: {doc.date_published.date()}"
        )


def _check_summary(doc: Document, result: ValidationResult) -> None:
    """
    Summary should be either empty (acceptable) or meaningfully long.
    A 1-2 word summary is likely a scraping artefact.
    """
    if doc.summary and len(doc.summary.strip()) < 10 and len(doc.summary.strip()) > 0:
        result.fail(f"Summary suspiciously short: '{doc.summary}'")


def _check_content_hash(doc: Document, result: ValidationResult) -> None:
    """Content hash must be a 64-character hex string (sha256)."""
    if not doc.content_hash or len(doc.content_hash) != 64:
        result.fail("Missing or invalid content hash")


# ── Batch deduplication ───────────────────────────────────────────────────────

def _check_duplicates(
    results: List[ValidationResult],
) -> List[ValidationResult]:
    """
    Mark duplicate URLs within a batch as failed.
    Keeps the first occurrence, flags the rest.
    """
    seen_urls:   set = set()
    seen_hashes: set = set()

    for result in results:
        doc = result.document

        if doc.url in seen_urls:
            result.fail(f"Duplicate URL in batch: '{doc.url}'")
        else:
            seen_urls.add(doc.url)

        if doc.content_hash and doc.content_hash in seen_hashes:
            result.fail(f"Duplicate content hash: '{doc.content_hash[:16]}...'")
        else:
            seen_hashes.add(doc.content_hash)

    return results


# ── Public API ────────────────────────────────────────────────────────────────

def validate_documents(
    documents: List[Document],
) -> Tuple[List[Document], List[ValidationResult]]:
    """
    Validate a list of documents.

    Returns:
        valid_docs : documents that passed all checks
        all_results: ValidationResult for every document (pass + fail)

    Usage:
        valid_docs, results = validate_documents(docs)
        for r in results:
            if not r.passed:
                print(r)
    """
    if not documents:
        logger.warning("validate_documents called with empty list")
        return [], []

    # run per-document checks
    results: List[ValidationResult] = []
    for doc in documents:
        result = ValidationResult(document=doc)
        _check_title(doc, result)
        _check_url(doc, result)
        _check_date(doc, result)
        _check_summary(doc, result)
        _check_content_hash(doc, result)
        results.append(result)

    # run batch deduplication
    results = _check_duplicates(results)

    # split into valid / invalid
    valid_docs = [r.document for r in results if r.passed]
    failed     = [r for r in results if not r.passed]

    # log summary
    logger.info(
        "Validation complete",
        extra={
            "total":   len(documents),
            "passed":  len(valid_docs),
            "failed":  len(failed),
        },
    )

    for r in failed:
        logger.warning(
            "Document failed validation",
            extra={
                "url":      r.document.url,
                "reasons":  r.failures,
            },
        )

    return valid_docs, results


def validation_report(results: List[ValidationResult]) -> dict:
    """
    Produce a summary dict from a list of ValidationResults.
    Used by pipeline/run.py to build catalog.json.
    """
    passed  = [r for r in results if r.passed]
    failed  = [r for r in results if not r.passed]

    failure_reasons: dict = {}
    for r in failed:
        for reason in r.failures:
            # bucket by first word of reason for easy grouping
            key = reason.split(":")[0].strip()
            failure_reasons[key] = failure_reasons.get(key, 0) + 1

    return {
        "total":           len(results),
        "passed":          len(passed),
        "failed":          len(failed),
        "pass_rate":       f"{len(passed)/max(len(results),1)*100:.1f}%",
        "failure_reasons": failure_reasons,
    }
