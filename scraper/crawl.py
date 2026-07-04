"""
scraper/crawl.py
----------------
MoSPI publications crawler.

The MoSPI website (mospi.gov.in) is a JavaScript single-page app whose
publication listing is served by a JSON backend API — plain HTML requests
only return an empty SPA shell. This crawler talks to that API directly,
then downloads each publication's PDF and extracts its text with pdfplumber.

Flow:
  1. fetch_publications()   → page through the real listing API
  2. parse_publication()    → turn each API item into a Document
  3. download + extract     → pull the PDF and read its text
  4. save to SQLite         → documents + files tables (with dedup)

Run:
    python -m scraper.crawl --max-pages 3
"""

import argparse
import hashlib
import io
from pathlib import Path
from typing import List, Optional, Tuple

import pdfplumber
import requests
from bs4 import BeautifulSoup

from scraper.config import settings
from scraper.db import init_db, save_document, save_file
from scraper.models import Document, PDFFile
from scraper.utils import (
    RateLimiter,
    get_logger,
    is_allowed,
    make_headers,
    normalize_category,
    normalize_date,
    normalize_text,
)

logger = get_logger(__name__)
limiter = RateLimiter()

# ── MoSPI backend API ─────────────────────────────────────────────────────────

SITE_BASE = "https://www.mospi.gov.in"
PUBLICATIONS_API = f"{SITE_BASE}/api/publications-reports/get-web-publications-report-list"

# Extraction bounds — keep the corpus rich but the run bounded.
MAX_PDF_MB = 30          # skip PDFs larger than this (avoid 100MB+ reports)
MAX_PDF_PAGES = 25       # only read the first N pages of each PDF
MAX_SUMMARY_CHARS = 45_000  # cap stored text per document


def _api_headers() -> dict:
    return {
        "User-Agent": settings.scraper_user_agent,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Origin": SITE_BASE,
        "Referer": f"{SITE_BASE}/publications-reports",
    }


def clean_title(raw: Optional[str]) -> str:
    """Strip the HTML wrapper MoSPI stores titles in (e.g. '<p>Title</p>')."""
    if not raw:
        return ""
    text = BeautifulSoup(raw, "html.parser").get_text(separator=" ")
    # Titles are single-line — collapse every whitespace run to one space.
    return " ".join(text.split())


# Map subject keywords in a title to a clean category slug.
_CATEGORY_KEYWORDS = [
    ("gdp", "gdp"), ("gross domestic", "gdp"), ("national income", "gdp"),
    ("national account", "gdp"), ("gva", "gdp"),
    ("consumer price", "cpi"), ("cpi", "cpi"), ("inflation", "cpi"),
    ("industrial production", "iip"), ("iip", "iip"),
    ("labour force", "plfs"), ("plfs", "plfs"), ("employment", "plfs"),
    ("labour", "labour"),
    ("consumption expenditure", "consumption"), ("hces", "consumption"),
    ("sustainable development", "sdg"), ("sdg", "sdg"),
    ("annual survey", "survey"), ("survey", "survey"),
    ("handbook", "handbook"), ("manual", "handbook"),
    ("statistical", "statistics"), ("yearbook", "statistics"),
]


def derive_category(title: str) -> str:
    """Pick a clean category slug from a title's subject keywords."""
    low = title.lower()
    for keyword, slug in _CATEGORY_KEYWORDS:
        if keyword in low:
            return slug
    return "publication"


def build_pdf_url(path: str) -> str:
    """Turn a relative API file path into an absolute download URL."""
    if not path:
        return ""
    if path.startswith("http"):
        return path
    return f"{SITE_BASE}/{path.lstrip('/')}"


# ── Listing API ───────────────────────────────────────────────────────────────

def fetch_publications(
    page_no: int,
    page_size: int,
    session: requests.Session,
) -> Tuple[List[dict], int]:
    """
    Fetch one page of publications from the MoSPI listing API.

    Returns:
        (items, total_pages) — items is the raw list of publication dicts.
        Returns ([], 0) on any error so the caller can stop gracefully.
    """
    payload = {
        "page_no": page_no,
        "page_size": page_size,
        "search_term": "",
        "sort_field": "published_year",
        "sort_order": "DESC",
        "from_date": "",
        "to_date": "",
        "lang": "en",
        "data_source": "web",
    }
    try:
        limiter.wait()
        resp = session.post(
            PUBLICATIONS_API, json=payload, headers=_api_headers(), timeout=30
        )
        resp.raise_for_status()
        body = resp.json()
        items = body.get("data", []) or []
        total_pages = int(body.get("pagination", {}).get("totalPages", 0) or 0)
        logger.info(
            "Fetched publications page",
            extra={"page": page_no, "items": len(items), "total_pages": total_pages},
        )
        return items, total_pages
    except Exception as exc:
        logger.error(
            "Failed to fetch publications page",
            extra={"page": page_no, "error": str(exc)},
        )
        return [], 0


def parse_publication(item: dict) -> Optional[Document]:
    """
    Convert one raw API publication item into a Document.

    Only items backed by a downloadable PDF (file_one) are returned — those
    are the ones we can extract real text from. Metadata-only entries
    (redirect links, no file) are skipped.
    """
    title = clean_title(item.get("title"))
    if not title or len(title) <= 3:
        return None

    file_one = item.get("file_one") or {}
    pdf_path = file_one.get("path", "") if isinstance(file_one, dict) else ""
    if not pdf_path or not pdf_path.lower().endswith(".pdf"):
        logger.debug("Skipping item without PDF", extra={"title": title[:60]})
        return None

    pdf_url = build_pdf_url(pdf_path)
    date_published = normalize_date(item.get("published_year"))
    category = derive_category(title)

    return Document(
        url=pdf_url,
        title=title,
        date_published=date_published,
        category=category,
        file_links=[pdf_url],
    )


# ── PDF text extraction ───────────────────────────────────────────────────────

def _compute_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_pdf(url: str, session: requests.Session) -> Optional[bytes]:
    """Download a PDF into memory, skipping oversized or non-PDF responses."""
    try:
        limiter.wait()
        resp = session.get(
            url, headers=make_headers(), timeout=90, stream=True
        )
        resp.raise_for_status()

        size = int(resp.headers.get("Content-Length", 0) or 0)
        if size and size > MAX_PDF_MB * 1024 * 1024:
            logger.warning(
                "Skipping oversized PDF",
                extra={"url": url, "mb": round(size / 1024 / 1024, 1)},
            )
            return None

        data = resp.content
        if len(data) < 1000:
            logger.warning("PDF too small — likely not a real PDF", extra={"url": url})
            return None
        return data
    except requests.RequestException as exc:
        logger.error("PDF download failed", extra={"url": url, "error": str(exc)})
        return None


def extract_pdf_text(data: bytes) -> Tuple[str, int]:
    """Extract text from the first MAX_PDF_PAGES pages of a PDF byte stream."""
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            total_pages = len(pdf.pages)
            pages = []
            for page in pdf.pages[:MAX_PDF_PAGES]:
                raw = page.extract_text()
                if raw:
                    pages.append(normalize_text(raw))
            text = "\n\n".join(pages)
        return text, total_pages
    except Exception as exc:
        logger.error("PDF text extraction failed", extra={"error": str(exc)})
        return "", 0


def _save_pdf_bytes(data: bytes, title: str) -> Path:
    """Persist the raw PDF under data/raw/pdf for provenance."""
    download_dir = Path(settings.pdf_download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    slug = normalize_category(title)[:60] or "document"
    path = download_dir / f"{slug}_{_compute_hash(data)[:8]}.pdf"
    path.write_bytes(data)
    return path


def ingest_document(doc: Document, session: requests.Session) -> str:
    """
    Download a document's PDF, extract its text, and persist everything.

    Returns one of: "saved", "skipped" (duplicate), "failed" (no text).
    """
    data = download_pdf(doc.url, session)
    if not data:
        return "failed"

    text, n_pages = extract_pdf_text(data)
    if not text or len(text) < 200:
        logger.warning("No usable text extracted", extra={"url": doc.url})
        return "failed"

    doc.summary = text[:MAX_SUMMARY_CHARS]

    doc_id = save_document(doc)
    if not doc_id:
        return "skipped"  # duplicate (hash already present)

    local_path = _save_pdf_bytes(data, doc.title)
    save_file(
        PDFFile(
            document_id=doc_id,
            file_url=doc.url,
            file_path=str(local_path),
            file_hash=_compute_hash(data),
            pages=n_pages,
        )
    )
    logger.info(
        "Document ingested",
        extra={"title": doc.title[:50], "pages": n_pages, "chars": len(text)},
    )
    return "saved"


# ── Orchestration ─────────────────────────────────────────────────────────────

def crawl(
    seed_urls: Optional[List[str]] = None,
    max_pages: int = settings.scraper_max_pages,
    page_size: int = 10,
) -> dict:
    """
    Crawl the MoSPI publications API and ingest each publication's PDF.

    Args:
        seed_urls : accepted for CLI/back-compat; the API endpoint is fixed.
        max_pages : number of listing pages to fetch (page_size items each).
        page_size : publications per listing page.

    Returns a summary dict of counts.
    """
    init_db()

    if not is_allowed(PUBLICATIONS_API):
        logger.warning("Crawling disallowed by robots.txt", extra={"url": PUBLICATIONS_API})
        return {"total_fetched": 0, "total_saved": 0, "total_skipped": 0,
                "total_failed": 0, "pages_crawled": 0}

    total_saved = total_skipped = total_failed = total_fetched = 0
    pages_crawled = 0

    with requests.Session() as session:
        for page_no in range(1, max_pages + 1):
            items, total_pages = fetch_publications(page_no, page_size, session)
            if not items:
                break
            pages_crawled += 1

            for item in items:
                doc = parse_publication(item)
                if not doc:
                    continue
                total_fetched += 1
                outcome = ingest_document(doc, session)
                if outcome == "saved":
                    total_saved += 1
                elif outcome == "skipped":
                    total_skipped += 1
                else:
                    total_failed += 1

            if total_pages and page_no >= total_pages:
                break

    summary = {
        "total_fetched": total_fetched,
        "total_saved": total_saved,
        "total_skipped": total_skipped,
        "total_failed": total_failed,
        "pages_crawled": pages_crawled,
    }
    logger.info("Crawl complete", extra=summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl MoSPI publications and extract PDF text.")
    parser.add_argument("--seed-url", nargs="+", default=None,
                        help="Accepted for compatibility; the API endpoint is fixed.")
    parser.add_argument("--max-pages", type=int, default=settings.scraper_max_pages)
    parser.add_argument("--page-size", type=int, default=10)
    args = parser.parse_args()

    summary = crawl(
        seed_urls=args.seed_url,
        max_pages=args.max_pages,
        page_size=args.page_size,
    )

    print("\n── Crawl Summary ───────────────────────────────")
    print(f"  Publications found : {summary['total_fetched']}")
    print(f"  Ingested (saved)   : {summary['total_saved']}")
    print(f"  Duplicates skipped : {summary['total_skipped']}")
    print(f"  Failed (no text)   : {summary['total_failed']}")
    print(f"  Pages crawled      : {summary['pages_crawled']}")
    print("────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
