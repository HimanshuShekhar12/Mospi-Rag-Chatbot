"""
scraper/parse.py
----------------
PDF downloader and content extractor for the MoSPI scraper.

Downloads PDFs linked to each document, extracts text and tables
using pdfplumber, computes file hash for deduplication, and saves
everything to SQLite.

Entry point:
    python -m scraper.parse
"""

import argparse
import hashlib
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pdfplumber
import requests

from scraper.config import settings
from scraper.db import get_all_documents, get_connection, init_db, save_file, save_table
from scraper.models import Document, ExtractedTable, PDFFile
from scraper.utils import RateLimiter, get_logger, make_headers, normalize_text

logger  = get_logger(__name__)
limiter = RateLimiter()


def _compute_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def _safe_filename(url: str) -> str:
    name = url.split("/")[-1].split("?")[0]
    name = "".join(c for c in name if c.isalnum() or c in "._-")
    return name or "document.pdf"


def _is_already_downloaded(file_hash: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id FROM files WHERE file_hash = ?", (file_hash,)
        ).fetchone()
    return row is not None


def _get_pdf_urls_from_db() -> List[Tuple[int, str]]:
    """
    Get (document_id, pdf_url) pairs.
    For our hardcoded documents, the URL itself is the PDF.
    Also checks file_links column if it exists.
    """
    pairs = []
    with get_connection() as conn:
        # get all documents
        rows = conn.execute(
            "SELECT id, url FROM documents"
        ).fetchall()

        for row in rows:
            doc_id = row["id"]
            url    = row["url"]

            # if the document URL itself is a PDF
            if url.lower().endswith(".pdf"):
                pairs.append((doc_id, url))
                continue

            # check file_links column if exists
            cols = [r[1] for r in conn.execute("PRAGMA table_info(documents)").fetchall()]
            if "file_links" in cols:
                fl_row = conn.execute(
                    "SELECT file_links FROM documents WHERE id = ?", (doc_id,)
                ).fetchone()
                if fl_row and fl_row["file_links"]:
                    for link in fl_row["file_links"].split("\n"):
                        link = link.strip()
                        if link and link.lower().endswith(".pdf"):
                            pairs.append((doc_id, link))

    logger.info("Found PDF URLs", extra={"count": len(pairs)})
    return pairs


def download_pdf(url: str, session: requests.Session, download_dir: Path) -> Optional[Tuple[Path, bytes]]:
    limiter.wait()
    try:
        response = session.get(url, headers=make_headers(), timeout=30, stream=True)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "html" in content_type.lower() and not url.lower().endswith(".pdf"):
            logger.warning("Skipping non-PDF URL", extra={"url": url})
            return None

        file_bytes = response.content
        if len(file_bytes) < 1000:
            logger.warning("File too small — likely not a real PDF", extra={"url": url, "bytes": len(file_bytes)})
            return None

        filename   = _safe_filename(url)
        local_path = download_dir / filename

        if local_path.exists():
            prefix     = _compute_hash(file_bytes)[:8]
            local_path = download_dir / f"{prefix}_{filename}"

        local_path.write_bytes(file_bytes)
        logger.info("PDF downloaded", extra={"url": url, "path": str(local_path), "bytes": len(file_bytes)})
        return local_path, file_bytes

    except requests.RequestException as exc:
        logger.error("Failed to download PDF", extra={"url": url, "error": str(exc)})
        return None


def extract_text(pdf_path: Path) -> Tuple[str, int]:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages_text = []
            for page in pdf.pages:
                raw = page.extract_text()
                if raw:
                    pages_text.append(normalize_text(raw))
            full_text  = "\n\n".join(pages_text)
            page_count = len(pdf.pages)
        logger.info("Text extracted", extra={"path": str(pdf_path), "pages": page_count, "chars": len(full_text)})
        return full_text, page_count
    except Exception as exc:
        logger.error("Text extraction failed", extra={"path": str(pdf_path), "error": str(exc)})
        return "", 0


def extract_tables(pdf_path: Path) -> List[Tuple[int, List[List]]]:
    results = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                if not tables:
                    continue
                for table in tables:
                    clean = [
                        [normalize_text(str(cell)) if cell else "" for cell in row]
                        for row in table
                        if any(cell for cell in row)
                    ]
                    if len(clean) >= 2 and len(clean[0]) >= 2:
                        results.append((page_num, clean))
    except Exception as exc:
        logger.error("Table extraction failed", extra={"path": str(pdf_path), "error": str(exc)})
    return results


def _update_document_summary(document_id: int, text: str) -> None:
    snippet = text[:500].strip()
    if not snippet:
        return
    with get_connection() as conn:
        conn.execute(
            "UPDATE documents SET summary = ? WHERE id = ? AND (summary IS NULL OR summary = '')",
            (snippet, document_id),
        )


def parse_all(max_pdfs: Optional[int] = None) -> dict:
    init_db()

    download_dir = Path(settings.pdf_download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    pdf_pairs = _get_pdf_urls_from_db()
    if not pdf_pairs:
        logger.warning("No PDF URLs found in database")
        return {"error": "no PDF URLs found"}

    total_downloaded = 0
    total_skipped    = 0
    total_failed     = 0
    total_tables     = 0
    pdf_count        = 0

    with requests.Session() as session:
        for doc_id, pdf_url in pdf_pairs:
            if max_pdfs and pdf_count >= max_pdfs:
                break

            result = download_pdf(pdf_url, session, download_dir)
            if not result:
                total_failed += 1
                continue

            local_path, file_bytes = result
            file_hash = _compute_hash(file_bytes)

            if _is_already_downloaded(file_hash):
                logger.debug("Duplicate PDF skipped", extra={"url": pdf_url})
                total_skipped += 1
                continue

            full_text, page_count = extract_text(local_path)

            if full_text:
                _update_document_summary(doc_id, full_text)

            pdf_file = PDFFile(
                document_id=doc_id,
                file_url=pdf_url,
                file_path=str(local_path),
                file_hash=file_hash,
                pages=page_count,
            )
            file_id = save_file(pdf_file)

            if not file_id:
                total_skipped += 1
                continue

            pdf_file.id = file_id
            total_downloaded += 1
            pdf_count += 1

            tables = extract_tables(local_path)
            for page_num, table_data in tables:
                table = ExtractedTable(
                    document_id=doc_id,
                    source_file_id=file_id,
                    page_number=page_num,
                    table_data=table_data,
                )
                save_table(table)
                total_tables += 1

            logger.info("PDF processed", extra={"url": pdf_url, "pages": page_count, "tables": len(tables)})

    summary = {
        "total_downloaded": total_downloaded,
        "total_skipped":    total_skipped,
        "total_failed":     total_failed,
        "total_tables":     total_tables,
    }
    logger.info("Parse complete", extra=summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PDFs and extract text + tables.")
    parser.add_argument("--max-pdfs", type=int, default=None)
    args = parser.parse_args()

    summary = parse_all(max_pdfs=args.max_pdfs)

    print("\n── Parse Summary ───────────────────────────────")
    print(f"  PDFs downloaded : {summary.get('total_downloaded', 0)}")
    print(f"  Duplicates skip : {summary.get('total_skipped', 0)}")
    print(f"  Failed          : {summary.get('total_failed', 0)}")
    print(f"  Tables extracted: {summary.get('total_tables', 0)}")
    print("────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
